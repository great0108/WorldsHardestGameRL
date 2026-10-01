from __future__ import annotations

"""NGU-inspired intrinsic exploration for the WHG pixel PPO trainer.

This is intentionally a small, self-contained "NGU-lite" component rather
than a full Agent57/NGU reproduction:

* an inverse-dynamics encoder learns features that emphasize controllable
  visual changes;
* a slowly moving target encoder provides a more stable embedding space;
* each vector-environment instance keeps its own episodic k-NN memory;
* the episodic pseudo-count follows NGU Algorithm 1 closely: normalized
  k-NN squared distances -> cluster threshold -> inverse kernel -> 1/s;
* novel states receive an intrinsic reward during training only.

The game/extrinsic reward is not changed.  Evaluation environments should not
be wrapped with this class.
"""

from collections import deque
import copy
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class _InverseDynamicsModel(nn.Module):
    def __init__(self, observation_space, n_actions: int, embedding_dim: int):
        super().__init__()
        if len(observation_space.shape) != 3:
            raise ValueError(
                "EpisodicNoveltyVecEnv expects channel-first image observations"
            )
        channels = int(observation_space.shape[0])
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float() / 255.0
            flat_dim = int(self.encoder(sample).shape[1])

        self.projection = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, int(embedding_dim)),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(2 * int(embedding_dim), 256),
            nn.ReLU(),
            nn.Linear(256, int(n_actions)),
        )

    def encode(self, obs: th.Tensor) -> th.Tensor:
        x = obs.float() / 255.0
        z = self.projection(self.encoder(x))
        return z

    def inverse_logits(self, obs_t: th.Tensor, obs_tp1: th.Tensor) -> th.Tensor:
        z_t = self.encode(obs_t)
        z_tp1 = self.encode(obs_tp1)
        return self.inverse_head(th.cat([z_t, z_tp1], dim=1))


class _TargetEncoder(nn.Module):
    def __init__(self, source: _InverseDynamicsModel):
        super().__init__()
        self.encoder = copy.deepcopy(source.encoder)
        self.projection = copy.deepcopy(source.projection)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = obs.float() / 255.0
        z = self.projection(self.encoder(x))
        return F.normalize(z, p=2, dim=1, eps=1e-8)


class EpisodicNoveltyVecEnv(VecEnvWrapper):
    """Add NGU-style episodic novelty reward to a training VecEnv.

    The wrapper is deliberately applied *outside* VecMonitor.  Therefore SB3's
    ``rollout/ep_rew_mean`` and the ``episode`` info continue to represent the
    original/extrinsic game reward, while PPO receives extrinsic + intrinsic.
    """

    def __init__(
        self,
        venv: VecEnv,
        *,
        beta: float = 0.01,
        embedding_dim: int = 64,
        memory_size: int = 512,
        k_neighbors: int = 10,
        inverse_lr: float = 1e-4,
        inverse_train_every: int = 4,
        inverse_batch_size: int = 128,
        target_tau: float = 0.001,
        cluster_distance: float = 0.008,
        kernel_epsilon: float = 1e-4,
        pseudo_count: float = 0.001,
        max_similarity: float = 8.0,
        device: str = "auto",
    ):
        super().__init__(venv)
        self.beta = float(beta)
        self.embedding_dim = int(embedding_dim)
        self.memory_size = int(memory_size)
        self.k_neighbors = int(k_neighbors)
        self.inverse_train_every = int(inverse_train_every)
        self.inverse_batch_size = int(inverse_batch_size)
        self.target_tau = float(target_tau)
        self.cluster_distance = float(cluster_distance)
        self.kernel_epsilon = float(kernel_epsilon)
        self.pseudo_count = float(pseudo_count)
        self.max_similarity = float(max_similarity)

        if self.beta < 0:
            raise ValueError("intrinsic beta must be >= 0")
        if self.embedding_dim < 2:
            raise ValueError("embedding_dim must be >= 2")
        if self.memory_size < 2:
            raise ValueError("memory_size must be >= 2")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if self.inverse_train_every < 1:
            raise ValueError("inverse_train_every must be >= 1")
        if self.inverse_batch_size < 1:
            raise ValueError("inverse_batch_size must be >= 1")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")
        if self.cluster_distance < 0.0:
            raise ValueError("cluster_distance must be >= 0")
        if self.kernel_epsilon <= 0.0:
            raise ValueError("kernel_epsilon must be > 0")
        if self.pseudo_count <= 0.0:
            raise ValueError("pseudo_count must be > 0")
        if self.max_similarity <= 0.0:
            raise ValueError("max_similarity must be > 0")

        if not hasattr(self.action_space, "n"):
            raise ValueError("NGU-lite currently requires a discrete action space")

        if device == "auto":
            device = "cuda" if th.cuda.is_available() else "cpu"
        self.device = th.device(device)

        self.model = _InverseDynamicsModel(
            self.observation_space,
            int(self.action_space.n),
            self.embedding_dim,
        ).to(self.device)
        self.target_encoder = _TargetEncoder(self.model).to(self.device)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=float(inverse_lr))

        self._memories = [deque(maxlen=self.memory_size) for _ in range(self.num_envs)]
        self._last_obs: np.ndarray | None = None
        self._pending_actions: np.ndarray | None = None
        self._vec_steps = 0

        # NGU Algorithm 1 normalizes the k-NN squared distances by a running
        # mean d_m^2.  Keep one pooled statistic across all vector environments.
        # Updating it once per vector step avoids accidentally applying the
        # running update N_env times per environment step.
        self._distance_mean = 1.0
        self._distance_count = 0
        self._last_inverse_loss = 0.0

    @th.no_grad()
    def _embed(self, obs: np.ndarray) -> np.ndarray:
        tensor = th.as_tensor(obs, device=self.device)
        z = self.target_encoder(tensor)
        return z.cpu().numpy().astype(np.float32, copy=False)

    def _update_target(self) -> None:
        tau = self.target_tau
        with th.no_grad():
            online = list(self.model.encoder.parameters()) + list(self.model.projection.parameters())
            target = list(self.target_encoder.encoder.parameters()) + list(
                self.target_encoder.projection.parameters()
            )
            for target_p, online_p in zip(target, online, strict=True):
                target_p.mul_(1.0 - tau).add_(online_p, alpha=tau)

    def _train_inverse(
        self,
        obs_t: np.ndarray,
        obs_tp1: np.ndarray,
        actions: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        if self._vec_steps % self.inverse_train_every != 0:
            return

        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return
        if idx.size > self.inverse_batch_size:
            # Deterministic subsampling keeps runs reproducible given the same
            # vector-env ordering without introducing another RNG stream.
            positions = np.linspace(0, idx.size - 1, self.inverse_batch_size).astype(np.int64)
            idx = idx[positions]

        x0 = th.as_tensor(obs_t[idx], device=self.device)
        x1 = th.as_tensor(obs_tp1[idx], device=self.device)
        a = th.as_tensor(actions[idx], device=self.device, dtype=th.long).reshape(-1)

        self.model.train()
        logits = self.model.inverse_logits(x0, x1)
        loss = F.cross_entropy(logits, a)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        self._update_target()
        self._last_inverse_loss = float(loss.detach().cpu())

    def _knn_squared_distances(self, env_idx: int, z: np.ndarray) -> np.ndarray:
        """Return squared distances to the available k nearest episodic states."""
        memory = self._memories[env_idx]
        if not memory:
            return np.empty(0, dtype=np.float32)

        mem = np.asarray(memory, dtype=np.float32)
        d2 = np.sum((mem - z[None, :]) ** 2, axis=1)
        k = min(self.k_neighbors, int(d2.size))
        return np.partition(d2, k - 1)[:k].astype(np.float32, copy=False)

    def _update_distance_mean(self, distance_batches: list[np.ndarray]) -> None:
        """Update pooled running d_m^2 once for the whole vector step.

        NGU Algorithm 1 uses a running average of squared k-NN distances.
        A cumulative running mean is used here, matching the intent of the
        original pseudo-count normalization while remaining independent of the
        number of parallel environments.
        """
        usable = [d[np.isfinite(d)] for d in distance_batches if d.size]
        if not usable:
            return
        flat = np.concatenate(usable).astype(np.float64, copy=False)
        if flat.size == 0:
            return

        batch_count = int(flat.size)
        batch_mean = float(np.mean(flat))
        if self._distance_count == 0:
            self._distance_mean = batch_mean
            self._distance_count = batch_count
            return

        total = self._distance_count + batch_count
        self._distance_mean += (batch_mean - self._distance_mean) * (batch_count / total)
        self._distance_count = total

    def _episodic_reward_from_distances(
        self, nearest_d2: np.ndarray
    ) -> tuple[float, float]:
        """Compute NGU Algorithm-1 episodic pseudo-count reward.

        d_n = max(d_k / d_m^2 - xi, 0)
        K   = epsilon / (d_n + epsilon)
        s   = sqrt(sum(K)) + c
        r   = 0 if s > s_m else 1 / s
        """
        if nearest_d2.size == 0:
            # No pseudo-count exists yet.  This also avoids an artificial huge
            # 1/c reward on the first state of every episode.
            return 0.0, 0.0

        distance_mean = max(float(self._distance_mean), 1e-8)
        normalized = nearest_d2.astype(np.float64, copy=False) / distance_mean
        clustered = np.maximum(normalized - self.cluster_distance, 0.0)
        kernels = self.kernel_epsilon / (clustered + self.kernel_epsilon)
        similarity = float(np.sqrt(np.sum(kernels)) + self.pseudo_count)

        if not np.isfinite(similarity) or similarity > self.max_similarity:
            return 0.0, similarity
        return float(1.0 / max(similarity, 1e-12)), similarity

    def _transition_observations(
        self, new_obs: np.ndarray, dones: np.ndarray, infos: list[dict]
    ) -> np.ndarray:
        out = np.array(new_obs, copy=True)
        for i, done in enumerate(dones):
            if done and "terminal_observation" in infos[i]:
                out[i] = np.asarray(infos[i]["terminal_observation"], dtype=out.dtype)
        return out

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self._last_obs = np.array(obs, copy=True)
        self._pending_actions = None
        for memory in self._memories:
            memory.clear()
        z = self._embed(obs)
        for i in range(self.num_envs):
            self._memories[i].append(z[i].astype(np.float16))
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions).copy()
        self.venv.step_async(actions)

    def step_wait(self):
        new_obs, extrinsic_rewards, dones, infos = self.venv.step_wait()
        if self._last_obs is None or self._pending_actions is None:
            raise RuntimeError("EpisodicNoveltyVecEnv.step_wait called before reset/step_async")

        transition_next = self._transition_observations(new_obs, dones, infos)
        self._vec_steps += 1

        # Death/fade frames are visually novel but uncontrollable.  They should
        # neither train the inverse model nor earn novelty reward.
        controllable = np.asarray(
            [
                bool(info.get("move_ready", True))
                and not bool(info.get("death_triggered", False))
                and not bool(info.get("death_completed", False))
                for info in infos
            ],
            dtype=np.bool_,
        )
        self._train_inverse(
            self._last_obs,
            transition_next,
            self._pending_actions,
            controllable,
        )

        z_next = self._embed(transition_next)

        # First collect every environment's k-NN distances.  Then update the
        # shared d_m^2 statistic ONCE for this vector step, as opposed to once
        # per environment.  All environments in the same vector step therefore
        # use the same normalization scale.
        knn_distances: list[np.ndarray] = []
        for i in range(self.num_envs):
            if controllable[i]:
                knn_distances.append(self._knn_squared_distances(i, z_next[i]))
            else:
                knn_distances.append(np.empty(0, dtype=np.float32))
        self._update_distance_mean(knn_distances)

        bonuses = np.zeros(self.num_envs, dtype=np.float32)
        similarities = np.zeros(self.num_envs, dtype=np.float32)
        knn_mean_d2 = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(self.num_envs):
            nearest = knn_distances[i]
            if controllable[i] and nearest.size:
                bonus, similarity = self._episodic_reward_from_distances(nearest)
                bonuses[i] = bonus
                similarities[i] = similarity
                knn_mean_d2[i] = float(np.mean(nearest))

            # Even zero-bonus respawn states enter memory so repeatedly taking
            # the same route after a death quickly becomes familiar.
            self._memories[i].append(z_next[i].astype(np.float16))

        train_rewards = np.asarray(extrinsic_rewards, dtype=np.float32) + self.beta * bonuses

        for i, info in enumerate(infos):
            info["reward_extrinsic"] = float(extrinsic_rewards[i])
            info["reward_intrinsic"] = float(bonuses[i])
            info["reward_intrinsic_scaled"] = float(self.beta * bonuses[i])
            info["reward_train_total"] = float(train_rewards[i])
            info["intrinsic_similarity"] = float(similarities[i])
            info["intrinsic_knn_mean_d2"] = float(knn_mean_d2[i])
            info["intrinsic_distance_mean"] = float(self._distance_mean)
            info["intrinsic_inverse_loss"] = float(self._last_inverse_loss)
            info["intrinsic_memory_size"] = int(len(self._memories[i]))

        # VecEnv auto-resets completed episodes.  Clear only those episodic
        # memories and seed them with the reset observation returned in new_obs.
        done_idx = np.flatnonzero(dones)
        if done_idx.size:
            z_reset = self._embed(new_obs[done_idx])
            for j, env_idx in enumerate(done_idx):
                self._memories[int(env_idx)].clear()
                self._memories[int(env_idx)].append(z_reset[j].astype(np.float16))

        self._last_obs = np.array(new_obs, copy=True)
        self._pending_actions = None
        return new_obs, train_rewards, dones, infos

    def save_intrinsic_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(
            {
                "model": self.model.state_dict(),
                "target_encoder": self.target_encoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "distance_mean": float(self._distance_mean),
                "distance_count": int(self._distance_count),
                "vec_steps": int(self._vec_steps),
                "config": {
                    "beta": self.beta,
                    "embedding_dim": self.embedding_dim,
                    "memory_size": self.memory_size,
                    "k_neighbors": self.k_neighbors,
                    "cluster_distance": self.cluster_distance,
                    "kernel_epsilon": self.kernel_epsilon,
                    "pseudo_count": self.pseudo_count,
                    "max_similarity": self.max_similarity,
                },
            },
            path,
        )

    def load_intrinsic_state(self, path: str | Path) -> None:
        payload = th.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.target_encoder.load_state_dict(payload["target_encoder"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        # Backward compatibility: old NGU-lite checkpoints stored a
        # ``distance_scale`` EMA instead of NGU's pooled d_m^2 statistic.
        self._distance_mean = float(
            payload.get("distance_mean", payload.get("distance_scale", 1.0))
        )
        self._distance_count = int(payload.get("distance_count", 0))
        self._vec_steps = int(payload.get("vec_steps", 0))
