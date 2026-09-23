from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from whg_swf_gym import WorldsHardestGameEnv
from whg_swf_gym.pixel_rl import PathProgressReward
from whg_swf_gym.navigation import ensure_spatial_oracle_cache
from whg_swf_gym.core import ACTION_TO_VECTOR
from whg_swf_gym.game_data import PLAYER_SPEED


RESET_CACHE_VERSION = 1


def _reset_cache_signature(*, level, width, height, resize_method):
    return {
        "version": RESET_CACHE_VERSION,
        "level": int(level),
        "width": int(width),
        "height": int(height),
        "resize_method": str(resize_method),
        "pixel_pipeline": "legacy_exact",
    }


def ensure_noop_reset_cache(
    cache_dir: str | Path,
    *,
    level: int,
    width: int,
    height: int,
    resize_method: str,
    max_delay: int,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Precompute every deterministic NOOP-start state and RGB frame once.

    The large RGB array is stored as a plain .npy file so spawned workers can
    open it with mmap_mode='r' instead of each keeping a private copy.
    """
    from whg_swf_gym.game_data import SOURCE_SHA1

    cache_dir = Path(cache_dir)
    frames_path = cache_dir / "frames.npy"
    states_path = cache_dir / "states.npz"
    manifest_path = cache_dir / "manifest.json"
    requested = _reset_cache_signature(
        level=level,
        width=width,
        height=height,
        resize_method=resize_method,
    )

    def valid_existing() -> bool:
        if force or not (manifest_path.exists() and frames_path.exists() and states_path.exists()):
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, value in requested.items():
                if manifest.get(key) != value:
                    return False
            if manifest.get("source_sha1") != SOURCE_SHA1:
                return False
            if int(manifest.get("max_delay", -1)) < int(max_delay):
                return False
            frames = np.load(frames_path, mmap_mode="r")
            ok = (
                frames.dtype == np.uint8
                and frames.ndim == 4
                and frames.shape[0] >= int(max_delay) + 1
                and tuple(frames.shape[1:]) == (int(height), int(width), 3)
            )
            del frames
            if not ok:
                return False
            with np.load(states_path, allow_pickle=False) as states:
                return (
                    len(states["valid"]) >= int(max_delay) + 1
                    and int(states["level"].item()) == int(level)
                    and str(states["source_sha1"].item()) == SOURCE_SHA1
                )
        except Exception:
            return False

    if valid_existing():
        if verbose:
            print(f"NOOP reset cache:     {cache_dir} (reused, mmap RGB frames)")
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_frames = cache_dir / "frames.tmp.npy"
    tmp_states = cache_dir / "states.tmp.npz"
    tmp_manifest = cache_dir / "manifest.tmp.json"
    for tmp in (tmp_frames, tmp_states, tmp_manifest):
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass

    if verbose:
        print(
            f"Building NOOP reset cache for level {level}: delays 0..{max_delay} "
            f"({width}x{height}, legacy_exact/{resize_method})..."
        )

    base = WorldsHardestGameEnv(
        level=int(level),
        observation_mode="pixels",
        reward_mode="zero",
        death_ends_episode=True,
        max_steps=max(1, int(max_delay) + 1),
        pixel_size=(int(width), int(height)),
        pixel_resize_method=resize_method,
    )
    n = int(max_delay) + 1
    num_coins = len(base.core.collected)
    frames = np.lib.format.open_memmap(
        tmp_frames,
        mode="w+",
        dtype=np.uint8,
        shape=(n, int(height), int(width), 3),
    )
    frames[:] = 0

    valid = np.zeros(n, dtype=np.bool_)
    player_x = np.zeros(n, dtype=np.float64)
    player_y = np.zeros(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)
    move_ready = np.zeros(n, dtype=np.bool_)
    deaths = np.zeros(n, dtype=np.int32)
    enemy_frame = np.zeros(n, dtype=np.int64)
    current_check = np.full(n, "", dtype="<U16")
    coin_mask = np.zeros((n, num_coins), dtype=np.bool_)
    level_cleared = np.zeros(n, dtype=np.bool_)
    game_cleared = np.zeros(n, dtype=np.bool_)
    total_frames = np.zeros(n, dtype=np.int64)
    level_frames = np.zeros(n, dtype=np.int64)
    base_steps = np.zeros(n, dtype=np.int64)

    def save_state(delay: int, obs: np.ndarray) -> None:
        c = base.core
        frames[delay] = np.asarray(obs, dtype=np.uint8)
        valid[delay] = True
        player_x[delay] = c.player_x
        player_y[delay] = c.player_y
        alpha[delay] = c.alpha
        move_ready[delay] = c.move_ready
        deaths[delay] = c.deaths
        enemy_frame[delay] = c.enemy_frame
        current_check[delay] = c.current_check
        if num_coins:
            coin_mask[delay, :] = np.asarray(c.collected, dtype=np.bool_)
        level_cleared[delay] = c.level_cleared
        game_cleared[delay] = c.game_cleared
        total_frames[delay] = c.total_frames
        level_frames[delay] = c.level_frames
        base_steps[delay] = base._steps

    try:
        obs, _ = base.reset(seed=0)
        save_state(0, obs)
        first_invalid = None
        for delay in range(1, n):
            obs, _reward, terminated, truncated, _info = base.step(0)
            if terminated or truncated:
                first_invalid = delay
                break
            save_state(delay, obs)

        frames.flush()
        del frames

        np.savez(
            tmp_states,
            level=np.asarray(int(level), dtype=np.int16),
            source_sha1=np.asarray(SOURCE_SHA1),
            valid=valid,
            player_x=player_x,
            player_y=player_y,
            alpha=alpha,
            move_ready=move_ready,
            deaths=deaths,
            enemy_frame=enemy_frame,
            current_check=current_check,
            coin_mask=coin_mask,
            level_cleared=level_cleared,
            game_cleared=game_cleared,
            total_frames=total_frames,
            level_frames=level_frames,
            base_steps=base_steps,
        )

        manifest = dict(requested)
        manifest.update({
            "source_sha1": SOURCE_SHA1,
            "max_delay": int(max_delay),
            "num_coins": int(num_coins),
            "valid_delays": int(valid.sum()),
            "first_invalid_delay": None if first_invalid is None else int(first_invalid),
            "frames_shape": [n, int(height), int(width), 3],
            "frames_dtype": "uint8",
        })
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Replace only after all pieces were written successfully.
        tmp_frames.replace(frames_path)
        tmp_states.replace(states_path)
        tmp_manifest.replace(manifest_path)
    finally:
        base.close()
        try:
            del frames
        except Exception:
            pass

    if verbose:
        mb = frames_path.stat().st_size / (1024 * 1024)
        print(
            f"NOOP reset cache ready: valid={int(valid.sum())}/{n}, "
            f"RGB={mb:.2f} MiB mmap, cache={cache_dir}"
        )
    return cache_dir


class _LeanCloudpickleWrapper:
    """Cloudpickle env factories without importing Stable-Baselines3 in workers."""

    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, state):
        import cloudpickle
        self.fn = cloudpickle.loads(state)


def _lean_subproc_worker(remote, parent_remote, env_fn_wrapper):
    """Minimal Gymnasium worker with optional shared-memory observations.

    IMPORTANT: this function intentionally imports no stable_baselines3/torch.

    In shared-observation mode, RGB stacks never travel through the Pipe.
    Workers write observations into one of two ping-pong shared-memory banks;
    a third bank stores terminal observations for VecEnv/SB3 semantics.
    """
    parent_remote.close()
    env = env_fn_wrapper.fn()
    reset_info = {}

    obs_shm = None
    obs_banks = None
    terminal_obs = None

    def publish_obs(obs, bank: int) -> None:
        if obs_banks is None:
            return
        np.copyto(obs_banks[int(bank)], np.asarray(obs), casting="no")

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == "set_obs_shm":
                # Attach only after the parent has learned observation shape/dtype.
                # The parent owns unlink(); children only close their handles.
                from multiprocessing import shared_memory
                import os

                name, env_index, n_envs, obs_shape, dtype_str = data
                try:
                    # Python 3.13+: do not register child attachments with the
                    # resource tracker because the parent owns the segment.
                    obs_shm = shared_memory.SharedMemory(name=name, track=False)
                except TypeError:
                    # Python 3.11/3.12 compatibility.
                    obs_shm = shared_memory.SharedMemory(name=name)
                    if os.name != "nt":
                        # On POSIX, prevent each spawned child resource tracker
                        # from unlinking a segment owned by the parent.
                        try:
                            from multiprocessing import resource_tracker
                            resource_tracker.unregister(obs_shm._name, "shared_memory")
                        except Exception:
                            pass

                obs_shape = tuple(int(v) for v in obs_shape)
                dtype = np.dtype(dtype_str)
                one_obs_bytes = int(np.prod(obs_shape, dtype=np.int64)) * dtype.itemsize
                current0_offset = int(env_index) * one_obs_bytes
                current1_offset = (int(n_envs) + int(env_index)) * one_obs_bytes
                terminal_offset = (2 * int(n_envs) + int(env_index)) * one_obs_bytes

                obs_banks = (
                    np.ndarray(obs_shape, dtype=dtype, buffer=obs_shm.buf, offset=current0_offset),
                    np.ndarray(obs_shape, dtype=dtype, buffer=obs_shm.buf, offset=current1_offset),
                )
                terminal_obs = np.ndarray(
                    obs_shape, dtype=dtype, buffer=obs_shm.buf, offset=terminal_offset
                )
                remote.send(True)

            elif cmd == "step":
                if obs_banks is None:
                    action = data
                else:
                    action, target_bank = data

                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                info = dict(info)
                info["TimeLimit.truncated"] = bool(truncated and not terminated)

                if done:
                    if obs_banks is None:
                        info["terminal_observation"] = obs
                    else:
                        np.copyto(terminal_obs, np.asarray(obs), casting="no")
                        # Rehydrated to a normal ndarray in the parent before
                        # VecTransposeImage/VecMonitor see the info dict.
                        info["_terminal_observation_shm"] = True
                    obs, reset_info = env.reset()

                if obs_banks is None:
                    remote.send((obs, reward, done, info, reset_info))
                else:
                    publish_obs(obs, int(target_bank))
                    remote.send((reward, done, info, reset_info))

            elif cmd == "reset":
                if obs_banks is None:
                    seed, options = data
                else:
                    seed, options, target_bank = data
                kwargs = {"options": options} if options else {}
                obs, reset_info = env.reset(seed=seed, **kwargs)
                if obs_banks is None:
                    remote.send((obs, reset_info))
                else:
                    publish_obs(obs, int(target_bank))
                    remote.send(reset_info)

            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))

            elif cmd == "get_attr":
                name = data
                try:
                    value = env.get_wrapper_attr(name)
                except Exception:
                    value = getattr(env, name)
                remote.send(value)

            elif cmd == "has_attr":
                name = data
                try:
                    env.get_wrapper_attr(name)
                    remote.send(True)
                except Exception:
                    remote.send(hasattr(env, name))

            elif cmd == "set_attr":
                name, value = data
                setattr(env, name, value)
                remote.send(None)

            elif cmd == "env_method":
                method_name, method_args, method_kwargs = data
                try:
                    method = env.get_wrapper_attr(method_name)
                except Exception:
                    method = getattr(env, method_name)
                remote.send(method(*method_args, **method_kwargs))

            elif cmd == "render":
                remote.send(env.render())

            elif cmd == "close":
                remote.close()
                break

            else:
                raise NotImplementedError(f"Unknown worker command: {cmd}")

    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        if obs_shm is not None:
            try:
                obs_shm.close()
            except Exception:
                pass


class RandomNoopFrameStack(gym.Wrapper):
    """Random NOOP pre-roll plus per-worker frame stacking.

    The wrapper outputs H x W x (3*n_stack). VecTransposeImage later converts
    this to (3*n_stack) x H x W, matching the channel-first PPO model shape.

    Action 0 is WHG's "stay" action.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        n_stack: int,
        delay_min: int,
        delay_max: int,
        agent_max_steps: int,
        rng_seed: int,
        frame_skip: int = 1,
        reset_cache_dir: str | Path,
    ):
        super().__init__(env)
        self.n_stack = int(n_stack)
        self.delay_min = int(delay_min)
        self.delay_max = int(delay_max)
        # agent_max_steps remains a limit in real Flash frames.
        self.agent_max_steps = int(agent_max_steps)
        self.frame_skip = int(frame_skip)
        self.reset_cache_dir = Path(reset_cache_dir)
        self._rng = np.random.default_rng(int(rng_seed))

        if self.n_stack < 1:
            raise ValueError("n_stack must be >= 1")
        if self.delay_min < 0:
            raise ValueError("delay_min must be >= 0")
        if self.delay_max < self.delay_min:
            raise ValueError("delay_max must be >= delay_min")
        if self.agent_max_steps < 1:
            raise ValueError("agent_max_steps must be >= 1")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")

        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 3:
            raise ValueError(f"expected HWC Box image observation, got {space}")

        h, w, c = space.shape
        self._channels_per_frame = int(c)
        self._stack = np.zeros((h, w, c * self.n_stack), dtype=space.dtype)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([space.low] * self.n_stack, axis=-1),
            high=np.concatenate([space.high] * self.n_stack, axis=-1),
            dtype=space.dtype,
        )

        self._start_delay_frames = 0
        self._agent_flash_frames = 0   # real 30-FPS game frames

        # The training pipeline always places PathProgressReward directly below
        # this wrapper.
        if not isinstance(env, PathProgressReward):
            raise TypeError("RandomNoopFrameStack requires PathProgressReward")
        self._reward_wrapper = env
        self._base_env = env.unwrapped

        required_cached_attrs = ("core", "_info", "_steps")
        if not all(hasattr(self._base_env, name) for name in required_cached_attrs):
            raise RuntimeError(
                "cached reset requires WorldsHardestGameEnv internals: "
                + ", ".join(required_cached_attrs)
            )

        self._load_reset_cache()

        self._last_reset_ms = 0.0

    def _load_reset_cache(self) -> None:
        cache_dir = self.reset_cache_dir
        manifest_path = cache_dir / "manifest.json"
        frames_path = cache_dir / "frames.npy"
        states_path = cache_dir / "states.npz"
        if not (manifest_path.exists() and frames_path.exists() and states_path.exists()):
            raise FileNotFoundError(f"incomplete NOOP reset cache: {cache_dir}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("version", -1)) != RESET_CACHE_VERSION:
            raise ValueError("NOOP reset cache version mismatch")
        if int(manifest.get("level", -1)) != int(self._base_env.core.level):
            raise ValueError("NOOP reset cache level mismatch")
        if int(manifest.get("max_delay", -1)) < self.delay_max:
            raise ValueError(
                f"NOOP reset cache covers only 0..{manifest.get('max_delay')}, "
                f"but this worker needs up to {self.delay_max}"
            )

        frames = np.load(frames_path, mmap_mode="r")
        if tuple(frames.shape[1:]) != tuple(self.env.observation_space.shape):
            raise ValueError(
                f"NOOP reset frame shape {frames.shape[1:]} does not match "
                f"worker observation {self.env.observation_space.shape}"
            )

        state_keys = (
            "level", "valid", "player_x", "player_y", "alpha", "move_ready",
            "deaths", "enemy_frame", "current_check", "coin_mask",
            "level_cleared", "game_cleared", "total_frames", "level_frames",
            "base_steps",
        )
        with np.load(states_path, allow_pickle=False) as loaded:
            states = {key: np.asarray(loaded[key]) for key in state_keys}
        if int(states["level"].item()) != int(self._base_env.core.level):
            raise ValueError("NOOP reset state-cache level mismatch")
        if len(states["valid"]) != len(frames):
            raise ValueError("NOOP reset frames/states length mismatch")

        mask = np.asarray(states["valid"], dtype=np.bool_)
        if not np.any(mask[self.delay_min:self.delay_max + 1]):
            raise RuntimeError(
                f"NOOP reset cache has no valid delays in "
                f"{self.delay_min}..{self.delay_max}"
            )

        self._reset_frames = frames
        self._reset_states = states

    def _sample_cached_delay(self) -> int:
        """Match the old rejection-sampling RNG stream without simulating physics."""
        states = self._reset_states
        valid = np.asarray(states["valid"], dtype=np.bool_)
        for _ in range(100):
            delay = self._sample_delay()
            if delay < len(valid) and bool(valid[delay]):
                return int(delay)
        raise RuntimeError(
            "Could not sample a valid cached random-delay start after 100 attempts"
        )

    def _restore_cached_core_state(self, delay: int) -> None:
        states = self._reset_states
        base = self._base_env
        core = base.core
        if int(states["level"].item()) != int(core.level):
            raise RuntimeError("cached reset attempted to restore a different level")

        core.player_x = float(states["player_x"][delay])
        core.player_y = float(states["player_y"][delay])
        core.alpha = float(states["alpha"][delay])
        core.move_ready = bool(states["move_ready"][delay])
        core.deaths = int(states["deaths"][delay])
        core.enemy_frame = int(states["enemy_frame"][delay])
        core.current_check = str(states["current_check"][delay])
        core.collected = [bool(v) for v in states["coin_mask"][delay]]
        core.level_cleared = bool(states["level_cleared"][delay])
        core.game_cleared = bool(states["game_cleared"][delay])
        core.total_frames = int(states["total_frames"][delay])
        core.level_frames = int(states["level_frames"][delay])
        base._steps = int(states["base_steps"][delay])

    def _restore_cached_stack(self, delay: int) -> None:
        frames = self._reset_frames
        c = self._channels_per_frame
        self._stack.fill(0)
        # Match the old reset exactly: before enough history exists, older
        # stack slots remain zero rather than repeating frame 0.
        for slot in range(self.n_stack):
            frame_idx = delay - (self.n_stack - 1 - slot)
            if frame_idx >= 0:
                self._stack[..., slot*c:(slot+1)*c] = frames[frame_idx]

    def _reset_cached(self, *, seed=None, options=None):
        """O(1) reset: restore a precomputed NOOP state and mmap-backed frames."""
        if options:
            requested_level = int(options.get("level", self._base_env.core.level))
            if requested_level != int(self._base_env.core.level):
                raise ValueError(
                    "cached reset does not support changing level through reset(options=...)"
                )

        # WorldsHardestGameEnv has deterministic physics. Reproduce Gym's reset
        # seeding side effect without calling env.reset(), which would render.
        if seed is not None:
            from gymnasium.utils import seeding
            rng, actual_seed = seeding.np_random(seed)
            self._base_env._np_random = rng
            self._base_env._np_random_seed = actual_seed

        delay = self._sample_cached_delay()
        self._restore_cached_core_state(delay)
        self._restore_cached_stack(delay)

        out_info = dict(self._base_env._info())
        out_info.update(self._reward_wrapper.resync_after_external_steps())
        return delay, out_info

    def _push(self, obs: np.ndarray) -> None:
        c = self._channels_per_frame
        if self.n_stack > 1:
            self._stack[..., :-c] = self._stack[..., c:].copy()
        self._stack[..., -c:] = obs

    def _sample_delay(self) -> int:
        if self.delay_min == self.delay_max:
            return self.delay_min
        return int(self._rng.integers(self.delay_min, self.delay_max + 1))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        t0 = time.perf_counter()
        delay, out_info = self._reset_cached(seed=seed, options=options)

        self._start_delay_frames = delay
        self._agent_flash_frames = 0
        self._last_reset_ms = (time.perf_counter() - t0) * 1000.0
        return self._stack.copy(), out_info

    def step(self, action):
        """Repeat one policy action for frame_skip real Flash frames.

        Reward is accumulated per underlying frame, so changing frame_skip does
        not silently change the reward definition. Every intermediate RGB frame
        is pushed into the stack, preserving real motion history.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            # Keep --max-steps defined in real Flash frames.
            if self._agent_flash_frames >= self.agent_max_steps:
                truncated = True
                info = dict(info)
                info["agent_time_limit"] = True
                break

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            self._agent_flash_frames += 1
            self._push(obs)

            if (
                self._agent_flash_frames >= self.agent_max_steps
                and not terminated
                and not truncated
            ):
                truncated = True
                info = dict(info)
                info["agent_time_limit"] = True

            if terminated or truncated:
                break

        info = dict(info)
        info["start_delay_frames"] = self._start_delay_frames
        info["agent_flash_frames"] = self._agent_flash_frames
        info["reset_ms"] = self._last_reset_ms
        return self._stack.copy(), total_reward, terminated, truncated, info


def _make_callback_classes():
    # Child env workers never call this, so they never import SB3/torch.
    from stable_baselines3.common.callbacks import BaseCallback

    class WHGMetricsCallback(BaseCallback):
        """Log rolling success/death/progress/start-delay metrics."""

        def __init__(self, log_every_timesteps: int = 10_000, window: int = 100):
            super().__init__()
            self.log_every_timesteps = int(log_every_timesteps)
            self.last_log = 0
            self.successes = deque(maxlen=window)
            self.max_progress = deque(maxlen=window)
            self.start_delays = deque(maxlen=window)
            self.flash_lengths = deque(maxlen=window)
            self.reset_ms = deque(maxlen=window)
            self.deaths = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])

            for info in infos:
                if info.get("death_triggered", False):
                    self.deaths += 1

            for done, info in zip(dones, infos):
                if done:
                    self.successes.append(float(info.get("is_success", False)))
                    self.max_progress.append(float(info.get("max_path_progress", 0.0)))
                    self.start_delays.append(float(info.get("start_delay_frames", 0.0)))
                    self.flash_lengths.append(float(info.get("agent_flash_frames", 0.0)))
                    self.reset_ms.append(float(info.get("reset_ms", 0.0)))


            if self.num_timesteps - self.last_log >= self.log_every_timesteps:
                self.last_log = self.num_timesteps
                if self.successes:
                    self.logger.record("whg/success_rate_100", float(np.mean(self.successes)))
                if self.max_progress:
                    self.logger.record(
                        "whg/max_path_progress_100", float(np.mean(self.max_progress))
                    )
                if self.start_delays:
                    self.logger.record(
                        "whg/start_delay_frames_100", float(np.mean(self.start_delays))
                    )
                if self.flash_lengths:
                    self.logger.record(
                        "whg/ep_flash_frames_100", float(np.mean(self.flash_lengths))
                    )
                if self.reset_ms:
                    self.logger.record(
                        "whg/reset_ms_100", float(np.mean(self.reset_ms))
                    )
                self.logger.record("whg/deaths_total", float(self.deaths))
            return True


    class WHGEvalCallback(BaseCallback):
        """Deterministic policy evaluation over randomized start phases."""

        def __init__(self, eval_env, eval_every_timesteps: int, n_eval_episodes: int, save_dir: Path):
            super().__init__()
            self.eval_env = eval_env
            self.eval_every_timesteps = int(eval_every_timesteps)
            self.n_eval_episodes = int(n_eval_episodes)
            self.save_dir = Path(save_dir)
            self.last_eval = 0
            self.best_success = -1.0
            self.best_reward = -np.inf

        def _evaluate(self) -> tuple[float, float, float, float, float]:
            obs = self.eval_env.reset()
            returns, lengths, flash_lengths, successes, delays = [], [], [], [], []
            ep_return = 0.0
            ep_length = 0

            while len(returns) < self.n_eval_episodes:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                ep_return += float(rewards[0])
                ep_length += 1

                if bool(dones[0]):
                    info = infos[0]
                    returns.append(ep_return)
                    lengths.append(ep_length)
                    successes.append(float(info.get("is_success", False)))
                    delays.append(float(info.get("start_delay_frames", 0.0)))
                    flash_lengths.append(float(info.get("agent_flash_frames", 0.0)))
                    ep_return = 0.0
                    ep_length = 0

            return (
                float(np.mean(successes)),
                float(np.mean(returns)),
                float(np.mean(lengths)),
                float(np.mean(delays)),
                float(np.mean(flash_lengths)),
            )

        def _on_step(self) -> bool:
            if self.num_timesteps - self.last_eval < self.eval_every_timesteps:
                return True

            self.last_eval = self.num_timesteps
            (
                success,
                reward,
                length,
                mean_delay,
                mean_flash_length,
            ) = self._evaluate()
            self.logger.record("eval/success_rate", success)
            self.logger.record("eval/mean_reward", reward)
            self.logger.record("eval/mean_length", length)
            self.logger.record("eval/mean_flash_length", mean_flash_length)
            self.logger.record("eval/mean_start_delay_frames", mean_delay)

            print(
                f"[eval @ {self.num_timesteps:,}] success={success:.1%} "
                f"reward={reward:.3f} decisions={length:.1f} "
                f"flash_frames={mean_flash_length:.1f} mean_delay={mean_delay:.1f}f"
            )

            if success > self.best_success or (
                success == self.best_success and reward > self.best_reward
            ):
                self.best_success = success
                self.best_reward = reward
                self.model.save(str(self.save_dir / "best_model"))
                print(f"Saved new best model to {self.save_dir / 'best_model.zip'}")
            return True

        def _on_training_end(self) -> None:
            self.eval_env.close()

    return WHGMetricsCallback, WHGEvalCallback


def _enable_spatial_oracle_physics_fast_path(reward_env: PathProgressReward) -> None:
    """Replace per-frame wall hit-tests with the exact spatial-oracle transition.

    The oracle graph was built from the same authored 3 px action followed by
    the same wall-correction order as ``WorldsHardestGameCore``.  We reuse only
    the graph's *integer lattice delta* instead of assigning its float32 cached
    coordinates directly, preserving the core's exact runtime coordinates.

    If a runtime position is ever outside the oracle graph, the original
    ``_apply_action``/``_correct_walls`` path is used for that frame.
    """
    reward_env._ensure_oracle()
    oracle = reward_env.oracle
    if oracle is None:
        raise RuntimeError("PathProgressReward failed to initialize its spatial oracle")

    core = reward_env._core
    positions = np.asarray(oracle.positions, dtype=np.float64)
    transitions = np.asarray(oracle.transitions)
    dst_positions = positions[transitions]

    # Every graph edge must be an integer number of authored 3 px player steps.
    scaled_delta = (dst_positions - positions[:, None, :]) / float(PLAYER_SPEED)
    rounded_delta = np.rint(scaled_delta)
    if not np.allclose(scaled_delta, rounded_delta, rtol=0.0, atol=1e-4):
        raise RuntimeError("spatial oracle contains a non-lattice physics transition")
    delta_steps = rounded_delta.astype(np.int8, copy=False)

    action_steps = np.asarray(
        [ACTION_TO_VECTOR[action] for action in range(9)], dtype=np.int8
    )
    wall_corrected = np.any(delta_steps != action_steps[None, :, :], axis=-1)
    node_for_xy = oracle._node_for_xy

    original_apply_action = core._apply_action
    original_correct_walls = core._correct_walls
    no_fast_result = object()
    pending_wall_corrected = no_fast_result

    def fast_apply_action(action: int) -> None:
        nonlocal pending_wall_corrected
        node = node_for_xy.get(
            (round(float(core.player_x), 6), round(float(core.player_y), 6))
        )
        if node is None:
            pending_wall_corrected = no_fast_result
            original_apply_action(action)
            return

        dx_steps, dy_steps = delta_steps[int(node), int(action)]
        core.player_x += int(dx_steps) * float(PLAYER_SPEED)
        core.player_y += int(dy_steps) * float(PLAYER_SPEED)
        pending_wall_corrected = bool(wall_corrected[int(node), int(action)])

    def fast_correct_walls() -> bool:
        nonlocal pending_wall_corrected
        if pending_wall_corrected is no_fast_result:
            return bool(original_correct_walls())
        corrected = bool(pending_wall_corrected)
        pending_wall_corrected = no_fast_result
        return corrected

    core._apply_action = fast_apply_action
    core._correct_walls = fast_correct_walls

def prepare_runtime_caches(args, cache_root: str | Path, *, max_delay: int | None = None):
    """Create/reuse the reset and spatial-oracle caches used by every runtime."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    if max_delay is None:
        max_delay = max(int(args.delay_max), int(args.eval_delay_max))

    reset_cache_name = (
        f"reset_cache_level{args.level:02d}_"
        f"legacy_exact_{args.width}x{args.height}_{args.resize_method}"
    )
    bundled_reset_cache = Path(__file__).resolve().with_name(reset_cache_name)
    if getattr(args, "reset_cache", None):
        reset_cache_dir = Path(args.reset_cache)
    elif bundled_reset_cache.exists() and not getattr(args, "rebuild_reset_cache", False):
        reset_cache_dir = bundled_reset_cache
    else:
        reset_cache_dir = cache_root / reset_cache_name
    args.reset_cache = str(ensure_noop_reset_cache(
        reset_cache_dir,
        level=args.level,
        width=args.width,
        height=args.height,
        resize_method=args.resize_method,
        max_delay=max_delay,
        force=getattr(args, "rebuild_reset_cache", False),
        verbose=True,
    ).resolve())

    spatial_cache_name = f"spatial_oracle_level{args.level:02d}.npz"
    bundled_spatial_cache = Path(__file__).resolve().with_name(spatial_cache_name)
    if getattr(args, "spatial_cache", None):
        spatial_cache_path = Path(args.spatial_cache)
    elif bundled_spatial_cache.exists() and not getattr(args, "rebuild_spatial_cache", False):
        spatial_cache_path = bundled_spatial_cache
    else:
        spatial_cache_path = cache_root / spatial_cache_name
    args.spatial_cache = str(ensure_spatial_oracle_cache(
        spatial_cache_path,
        level=args.level,
        force=getattr(args, "rebuild_spatial_cache", False),
        verbose=True,
    ).resolve())
    return args


def make_worker(args, rank: int, *, eval_mode: bool = False):
    def _init():
        if eval_mode:
            delay_min = args.eval_delay_min
            delay_max = args.eval_delay_max
            rng_seed = args.seed + rank + 100_000
        else:
            delay_min = args.delay_min
            delay_max = args.delay_max
            rng_seed = args.seed + rank

        # Hidden pre-roll consumes real Flash frames, so give the core headroom.
        core_max_steps = args.max_steps + delay_max

        env = WorldsHardestGameEnv(
            level=args.level,
            observation_mode="pixels",
            reward_mode="zero",
            death_ends_episode=True,
            max_steps=core_max_steps,
            # legacy_exact keeps the exact old 550x400 rasterization + Pillow
            # resize, but performs the resize inside Renderer before converting
            # the full-size image to NumPy.
            pixel_size=(args.width, args.height),
                pixel_resize_method=args.resize_method,
        )
        env = PathProgressReward(
            env,
            cache_path=args.spatial_cache,
            step_penalty=args.step_penalty,
            progress_scale=args.progress_scale,
            death_penalty=args.death_penalty,
            checkpoint_bonus=args.checkpoint_bonus,
            coin_bonus=args.coin_bonus,
            clear_bonus=args.clear_bonus,
        )
        _enable_spatial_oracle_physics_fast_path(env)
        env = RandomNoopFrameStack(
            env,
            n_stack=args.frame_stack,
            delay_min=delay_min,
            delay_max=delay_max,
            agent_max_steps=args.max_steps,
            rng_seed=rng_seed,
            frame_skip=args.frame_skip,
            reset_cache_dir=args.reset_cache,
        )
        env.action_space.seed(rng_seed)
        env.observation_space.seed(rng_seed)
        return env

    return _init


def _make_lean_subproc_vec_env(env_fns, *, shared_observations: bool = True):
    """Create a spawn-light SB3 VecEnv.

    When ``shared_observations`` is enabled, workers place observations into a
    parent-owned shared-memory segment instead of pickling ~80 KB arrays through
    each Pipe every PPO decision. Two current-observation banks are alternated
    so SB3 can safely keep the previous observation until it adds it to the
    rollout buffer. A third bank holds terminal observations.
    """
    import multiprocessing as mp
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    class LeanSubprocVecEnv(VecEnv):
        def __init__(self, fns, use_shared_observations: bool):
            self.waiting = False
            self.closed = False
            self._ctx = mp.get_context("spawn")
            self._shared_observations = bool(use_shared_observations)
            self._obs_shm = None
            self._obs_shm_array = None
            self._current_obs_bank = 0
            self._pending_obs_bank = None
            n_envs = len(fns)

            pipes = [self._ctx.Pipe() for _ in range(n_envs)]
            self.remotes, self.work_remotes = zip(*pipes)
            self.processes = []

            for work_remote, remote, fn in zip(
                self.work_remotes, self.remotes, fns, strict=True
            ):
                process = self._ctx.Process(
                    target=_lean_subproc_worker,
                    args=(work_remote, remote, _LeanCloudpickleWrapper(fn)),
                    daemon=True,
                )
                process.start()
                self.processes.append(process)
                work_remote.close()

            self.remotes[0].send(("get_spaces", None))
            observation_space, action_space = self.remotes[0].recv()
            super().__init__(n_envs, observation_space, action_space)

            if self._shared_observations:
                from multiprocessing import shared_memory

                if not isinstance(observation_space, gym.spaces.Box):
                    raise TypeError(
                        "shared observations currently require a Box observation space"
                    )
                if observation_space.shape is None or observation_space.dtype is None:
                    raise TypeError("shared observations require fixed shape and dtype")

                obs_shape = tuple(int(v) for v in observation_space.shape)
                obs_dtype = np.dtype(observation_space.dtype)
                one_obs_elems = int(np.prod(obs_shape, dtype=np.int64))
                # bank0 + bank1 + terminal, each containing all envs contiguously.
                total_elems = 3 * n_envs * one_obs_elems
                self._obs_shm = shared_memory.SharedMemory(
                    create=True,
                    size=total_elems * obs_dtype.itemsize,
                )
                self._obs_shm_array = np.ndarray(
                    (3, n_envs, *obs_shape),
                    dtype=obs_dtype,
                    buffer=self._obs_shm.buf,
                )
                self._obs_shm_array.fill(0)

                payload_common = (
                    self._obs_shm.name,
                    n_envs,
                    obs_shape,
                    obs_dtype.str,
                )
                for env_index, remote in enumerate(self.remotes):
                    remote.send((
                        "set_obs_shm",
                        (
                            payload_common[0],
                            env_index,
                            payload_common[1],
                            payload_common[2],
                            payload_common[3],
                        ),
                    ))
                for remote in self.remotes:
                    if remote.recv() is not True:
                        raise RuntimeError("worker failed to attach observation shared memory")

        def _current_obs_view(self):
            # Do NOT copy here. Ping-pong banks guarantee this bank is not
            # overwritten until after SB3 has consumed it on the next iteration.
            return self._obs_shm_array[self._current_obs_bank]

        def step_async(self, actions):
            if self._shared_observations:
                target_bank = 1 - self._current_obs_bank
                self._pending_obs_bank = target_bank
                for remote, action in zip(self.remotes, actions, strict=True):
                    remote.send(("step", (action, target_bank)))
            else:
                for remote, action in zip(self.remotes, actions, strict=True):
                    remote.send(("step", action))
            self.waiting = True

        def step_wait(self):
            results = [remote.recv() for remote in self.remotes]
            self.waiting = False

            if not self._shared_observations:
                obs, rews, dones, infos, reset_infos = zip(*results, strict=True)
                self.reset_infos = list(reset_infos)
                return (
                    np.stack(obs),
                    np.asarray(rews, dtype=np.float32),
                    np.asarray(dones, dtype=np.bool_),
                    list(infos),
                )

            rews, dones, infos, reset_infos = zip(*results, strict=True)
            self.reset_infos = list(reset_infos)
            self._current_obs_bank = int(self._pending_obs_bank)
            self._pending_obs_bank = None

            # terminal_observation must remain valid independently of future
            # worker writes, so copy only those rare terminal frames.
            out_infos = []
            terminal_bank = self._obs_shm_array[2]
            for env_index, raw_info in enumerate(infos):
                info = dict(raw_info)
                if info.pop("_terminal_observation_shm", False):
                    info["terminal_observation"] = terminal_bank[env_index].copy()
                out_infos.append(info)

            return (
                self._current_obs_view(),
                np.asarray(rews, dtype=np.float32),
                np.asarray(dones, dtype=np.bool_),
                out_infos,
            )

        def reset(self):
            if self._shared_observations:
                # Reset into bank 0. The first environment step writes bank 1.
                self._current_obs_bank = 0
                self._pending_obs_bank = None
                for env_idx, remote in enumerate(self.remotes):
                    remote.send((
                        "reset",
                        (self._seeds[env_idx], self._options[env_idx], 0),
                    ))
                self.reset_infos = [remote.recv() for remote in self.remotes]
                self._reset_seeds()
                self._reset_options()
                return self._current_obs_view()

            for env_idx, remote in enumerate(self.remotes):
                remote.send(
                    ("reset", (self._seeds[env_idx], self._options[env_idx]))
                )
            results = [remote.recv() for remote in self.remotes]
            obs, reset_infos = zip(*results, strict=True)
            self.reset_infos = list(reset_infos)
            self._reset_seeds()
            self._reset_options()
            return np.stack(obs)

        def close(self):
            if self.closed:
                return

            if self.waiting:
                for remote in self.remotes:
                    try:
                        remote.recv()
                    except (EOFError, BrokenPipeError, OSError):
                        pass
                self.waiting = False

            for remote in self.remotes:
                try:
                    remote.send(("close", None))
                except (EOFError, BrokenPipeError, OSError):
                    pass

            for process in self.processes:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)

            if self._obs_shm is not None:
                try:
                    self._obs_shm.close()
                finally:
                    try:
                        self._obs_shm.unlink()
                    except FileNotFoundError:
                        pass
                self._obs_shm = None
                self._obs_shm_array = None

            self.closed = True

        def get_images(self):
            for remote in self.remotes:
                remote.send(("render", None))
            return [remote.recv() for remote in self.remotes]

        def has_attr(self, attr_name):
            for remote in self.remotes:
                remote.send(("has_attr", attr_name))
            return all(remote.recv() for remote in self.remotes)

        def _target_remotes(self, indices):
            idxs = self._get_indices(indices)
            return [self.remotes[i] for i in idxs]

        def get_attr(self, attr_name, indices=None):
            remotes = self._target_remotes(indices)
            for remote in remotes:
                remote.send(("get_attr", attr_name))
            return [remote.recv() for remote in remotes]

        def set_attr(self, attr_name, value, indices=None):
            remotes = self._target_remotes(indices)
            for remote in remotes:
                remote.send(("set_attr", (attr_name, value)))
            for remote in remotes:
                remote.recv()

        def env_method(
            self, method_name, *method_args, indices=None, **method_kwargs
        ):
            remotes = self._target_remotes(indices)
            payload = (method_name, method_args, method_kwargs)
            for remote in remotes:
                remote.send(("env_method", payload))
            return [remote.recv() for remote in remotes]

        def env_is_wrapped(self, wrapper_class, indices=None):
            # The child envs are WHG/Gymnasium wrappers, never SB3 Monitor.
            # Do not pickle an SB3 wrapper class into children, because that would
            # make Windows spawn import stable_baselines3 -> torch/CUDA there.
            idxs = self._get_indices(indices)
            return [False] * len(idxs)

    return LeanSubprocVecEnv(env_fns, shared_observations)


def finish_vec_wrapping(env):
    from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage

    env = VecMonitor(env)
    return VecTransposeImage(env)


def make_train_vec_env(args):
    workers = [make_worker(args, i) for i in range(args.n_envs)]

    if args.n_envs == 1:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv(workers)
    else:
        env = _make_lean_subproc_vec_env(
            workers,
            shared_observations=args.shared_observations,
        )

    return finish_vec_wrapping(env)


def make_eval_vec_env(args):
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([make_worker(args, 0, eval_mode=True)])
    return finish_vec_wrapping(env)

def main():
    # Heavy ML imports must stay inside main() on Windows.
    # Spawned environment workers re-import this training file, but they do not
    # execute main(), so they avoid loading PyTorch/CUDA/cuDNN.
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from model import WHGCNN

    WHGMetricsCallback, WHGEvalCallback = _make_callback_classes()

    ap = argparse.ArgumentParser(
        description="Pixel PPO with exact spatial geometry reward shaping"
    )
    ap.add_argument("--level", type=int, default=1, choices=range(1, 31))
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--n-envs", type=int, default=128)
    ap.add_argument(
        "--shared-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "for n-envs > 1, transfer observations through a parent-owned "
            "ping-pong shared-memory buffer instead of pickling them through Pipes; "
            "use --no-shared-observations for A/B comparison"
        ),
    )

    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="repeat each policy action for N real Flash frames",
    )
    ap.add_argument(
        "--reset-cache",
        default=None,
        help="NOOP reset cache directory; default: <out>/reset_cache_levelNN_...",
    )
    ap.add_argument(
        "--rebuild-reset-cache",
        action="store_true",
        help="force rebuilding the precomputed NOOP reset cache",
    )
    ap.add_argument("--width", type=int, default=110)
    ap.add_argument("--height", type=int, default=80)
    ap.add_argument("--resize-method", choices=["nearest", "bilinear", "box"], default="box")
    ap.add_argument(
        "--spatial-cache",
        default=None,
        help="optional exact-spatial .npz cache path; default: <out>/spatial_oracle_levelNN.npz",
    )
    ap.add_argument(
        "--rebuild-spatial-cache",
        action="store_true",
        help="force rebuilding the exact geometry-only spatial oracle cache",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="max agent-controlled Flash frames; pre-roll frames are extra",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=128, help="rollout steps per environment")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.9)
    ap.add_argument("--ent-coef", type=float, default=0.03)

    ap.add_argument("--step-penalty", type=float, default=-0.003)
    ap.add_argument("--progress-scale", type=float, default=0.05)
    ap.add_argument("--death-penalty", type=float, default=-1.0)
    ap.add_argument("--checkpoint-bonus", type=float, default=1.0)
    ap.add_argument("--coin-bonus", type=float, default=1.0)
    ap.add_argument("--clear-bonus", type=float, default=1.0)

    ap.add_argument(
        "--delay-min", type=int, default=0,
        help="minimum training NOOP pre-roll in Flash frames",
    )
    ap.add_argument(
        "--delay-max", type=int, default=100,
        help="maximum training NOOP pre-roll in Flash frames, inclusive",
    )
    ap.add_argument(
        "--eval-delay-min", type=int, default=0,
        help="minimum eval pre-roll in Flash frames",
    )
    ap.add_argument(
        "--eval-delay-max", type=int, default=100,
        help="maximum eval pre-roll in Flash frames",
    )

    ap.add_argument("--eval-every", type=int, default=200_000, help="0 disables evaluation")
    ap.add_argument(
        "--eval-episodes", type=int, default=10,
        help="deterministic eval episodes across randomized start phases",
    )
    ap.add_argument("--out", default="runs/level01")
    ap.add_argument(
        "--resume", default=None,
        help="saved PPO .zip; keep width/height/frame-stack identical",
    )
    args = ap.parse_args()

    if args.n_envs < 1:
        ap.error("--n-envs must be >= 1")
    if args.frame_stack < 1:
        ap.error("--frame-stack must be >= 1")
    if args.frame_skip < 1:
        ap.error("--frame-skip must be >= 1")
    if args.max_steps < 1:
        ap.error("--max-steps must be >= 1")
    if args.delay_min < 0:
        ap.error("--delay-min must be >= 0")
    if args.delay_max < args.delay_min:
        ap.error("--delay-max must be >= --delay-min")
    if args.batch_size > args.n_steps * args.n_envs:
        ap.error("--batch-size must be <= --n-steps * --n-envs")
    if args.eval_episodes < 1:
        ap.error("--eval-episodes must be >= 1")

    if args.eval_delay_min < 0:
        ap.error("--eval-delay-min must be >= 0")
    if args.eval_delay_max < args.eval_delay_min:
        ap.error("--eval-delay-max must be >= --eval-delay-min")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    prepare_runtime_caches(
        args,
        out,
        max_delay=max(args.delay_max, args.eval_delay_max),
    )

    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(
        f"Frame skip:           {args.frame_skip} "
        f"(one PPO decision = up to {args.frame_skip} Flash frames)"
    )
    print(
        f"Shared obs IPC:       {bool(args.shared_observations and args.n_envs > 1)} "
        f"({'ping-pong shared memory' if args.shared_observations and args.n_envs > 1 else 'Pipe/DummyVec'})"
    )
    print(f"Reset cache:          {args.reset_cache}")
    print(f"Pixel pipeline:       legacy_exact ({args.resize_method})")
    print("Reward shaping:       spatial")
    print(f"Spatial cache:        {args.spatial_cache}")
    print(
        f"Training start delay: {args.delay_min}..{args.delay_max} frames "
        f"({args.delay_min/30:.3f}..{args.delay_max/30:.3f}s)"
    )
    print(
        f"Evaluation delay:     {args.eval_delay_min}..{args.eval_delay_max} frames "
        f"({args.eval_delay_min/30:.3f}..{args.eval_delay_max/30:.3f}s)"
    )

    env = make_train_vec_env(args)
    print("Vector observation space:", env.observation_space)
    print(
        "Expected stacked shape: (%d, %d, %d)"
        % (3 * args.frame_stack, args.height, args.width)
    )

    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
        model.tensorboard_log = str(out / "tb")
        print(f"Resuming from {args.resume}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            n_epochs=4,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            policy_kwargs=dict(
                features_extractor_class=WHGCNN,
                features_extractor_kwargs=dict(
                    features_dim=2048,
                    channels=(64, 128, 128),
                ),
                share_features_extractor=False,

                net_arch=dict(
                    pi=[1024],
                    vf=[1024],
                ),
            ),
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(out / "tb"),
        )

    checkpoint_every = max(1, 1_000_000 // args.n_envs)
    callback_items = [
        WHGMetricsCallback(),
        CheckpointCallback(
            save_freq=checkpoint_every,
            save_path=str(out / "checkpoints"),
            name_prefix=f"level{args.level:02d}_pixels",
        ),
    ]

    if args.eval_every > 0:
        callback_items.append(
            WHGEvalCallback(
                make_eval_vec_env(args),
                eval_every_timesteps=args.eval_every,
                n_eval_episodes=args.eval_episodes,
                save_dir=out,
            )
        )

    callbacks = CallbackList(callback_items)

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            tb_log_name=f"level{args.level:02d}",
            reset_num_timesteps=not bool(args.resume),
            progress_bar=True,
        )
        model.save(str(out / "final_model"))
        print(f"Saved model to {out / 'final_model.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
