from __future__ import annotations

"""NGU-inspired intrinsic exploration for the WHG pixel PPO trainer.

This is intentionally a small, self-contained "NGU-lite" component rather
than a full Agent57/NGU reproduction:

* an inverse-dynamics encoder learns features that emphasize controllable
  visual changes;
* a slowly moving target encoder provides a more stable embedding space;
* each vector-environment instance keeps its own episodic k-NN memory;
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
        target_tau: float = 0.01,
        kernel_epsilon: float = 1e-3,
        distance_ema: float = 0.99,
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
        self.kernel_epsilon = float(kernel_epsilon)
        self.distance_ema_decay = float(distance_ema)

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
        if not 0.0 < self.distance_ema_decay < 1.0:
            raise ValueError("distance_ema must be in (0, 1)")

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
        self._distance_scale = 1.0
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

    def _novelty(self, env_idx: int, z: np.ndarray) -> float:
        memory = self._memories[env_idx]
        if not memory:
            return 1.0

        mem = np.asarray(memory, dtype=np.float32)
        d2 = np.sum((mem - z[None, :]) ** 2, axis=1)
        k = min(self.k_neighbors, int(d2.size))
        nearest = np.partition(d2, k - 1)[:k]
        mean_d2 = float(np.mean(nearest))

        # Normalize distances with a slowly moving global scale.  L2-normalized
        # embeddings make this well behaved even while the inverse encoder is
        # still learning.
        if np.isfinite(mean_d2) and mean_d2 > 1e-8:
            self._distance_scale = (
                self.distance_ema_decay * self._distance_scale
                + (1.0 - self.distance_ema_decay) * mean_d2
            )

        normalized = nearest / max(self._distance_scale, 1e-6)
        kernels = self.kernel_epsilon / (normalized + self.kernel_epsilon)
        similarity = float(np.sqrt(np.sum(kernels))) + 1e-3
        raw = 1.0 / similarity

        # Map an exact/repeated-state neighborhood to ~0 and very distant states
        # toward 1.  This prevents intrinsic reward from degenerating into a
        # generic positive living reward.
        close_floor = 1.0 / (np.sqrt(float(k)) + 1e-3)
        bonus = (raw - close_floor) / max(1.0 - close_floor, 1e-6)
        return float(np.clip(bonus, 0.0, 1.0))

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
        bonuses = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(self.num_envs):
            if controllable[i]:
                bonuses[i] = self._novelty(i, z_next[i])
            # Even zero-bonus respawn states enter memory so repeatedly taking
            # the same route after a death quickly becomes familiar.
            self._memories[i].append(z_next[i].astype(np.float16))

        train_rewards = np.asarray(extrinsic_rewards, dtype=np.float32) + self.beta * bonuses

        for i, info in enumerate(infos):
            info["reward_extrinsic"] = float(extrinsic_rewards[i])
            info["reward_intrinsic"] = float(bonuses[i])
            info["reward_intrinsic_scaled"] = float(self.beta * bonuses[i])
            info["reward_train_total"] = float(train_rewards[i])
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
                "distance_scale": float(self._distance_scale),
                "vec_steps": int(self._vec_steps),
                "config": {
                    "beta": self.beta,
                    "embedding_dim": self.embedding_dim,
                    "memory_size": self.memory_size,
                    "k_neighbors": self.k_neighbors,
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
        self._distance_scale = float(payload.get("distance_scale", 1.0))
        self._vec_steps = int(payload.get("vec_steps", 0))
