from __future__ import annotations

import gymnasium as gym
import numpy as np

from .navigation import ShortestPathOracle, current_navigation_objective


class PathProgressReward(gym.Wrapper):
    """Reward wrapper for pixel-only WHG agents.

    The agent still receives pixels only. The hidden geometry oracle is used
    exclusively to compute a geometry-only progress signal. The spatial oracle
    uses the game's exact action + wall-correction transition graph, including
    transient wall-overlap states. Permanently stationary enemies are treated as
    lethal static navigation obstacles. Moving enemies are deliberately excluded
    so avoiding them must be learned from the stacked image observations.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        step_penalty: float = -0.001,
        progress_scale: float = 0.02,
        death_penalty: float = -1.0,
        checkpoint_bonus: float = 2.0,
        coin_bonus: float = 1.0,
        clear_bonus: float = 10.0,
        cache_path: str | None = None,
        clip_progress: float | None = None,
    ):
        super().__init__(env)
        self.step_penalty = float(step_penalty)
        self.progress_scale = float(progress_scale)
        self.death_penalty = float(death_penalty)
        self.checkpoint_bonus = float(checkpoint_bonus)
        self.coin_bonus = float(coin_bonus)
        self.clear_bonus = float(clear_bonus)
        self.cache_path = cache_path
        self.clip_progress = None if clip_progress is None else float(clip_progress)
        self.oracle: ShortestPathOracle | None = None
        self._objective = None
        self._distance = 0.0
        self._segment_start_distance = 0.0
        self._best_distance = float("inf")

    @property
    def _core(self):
        return self.env.unwrapped.core

    def _ensure_oracle(self) -> None:
        if self.oracle is None or self.oracle.level != self._core.level:
            self.oracle = ShortestPathOracle(self._core, self.cache_path)

    def _measure(self):
        self._ensure_oracle()
        objective = current_navigation_objective(self._core)
        distance = self.oracle.distance(self._core.player_x, self._core.player_y, objective)
        return objective, distance

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Keep the static wall oracle across episode resets; rebuilding the
        # 3 px lattice every episode would dominate training time.
        if self.oracle is not None and self.oracle.level != self._core.level:
            self.oracle = None
        self._objective, self._distance = self._measure()
        self._segment_start_distance = self._distance
        self._best_distance = self._distance
        info = dict(info)
        info.update({
            "path_objective": self._objective.kind,
            "path_distance": self._distance,
            "path_progress": 0.0,
            "max_path_progress": 0.0,
            "is_success": False,
        })
        return obs, info

    def resync_after_external_steps(self) -> dict:
        """Resynchronize hidden shaping state after reset pre-roll bypasses step()."""
        if self.oracle is not None and self.oracle.level != self._core.level:
            self.oracle = None
        self._objective, self._distance = self._measure()
        self._segment_start_distance = self._distance
        self._best_distance = self._distance
        return {
            "path_objective": self._objective.kind,
            "path_distance": self._distance,
            "path_progress": 0.0,
            "max_path_progress": 0.0,
            "is_success": False,
        }

    def step(self, action):
        old_objective = self._objective
        old_distance = self._distance
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        new_objective, new_distance = self._measure()
        objective_changed = new_objective.key != old_objective.key

        progress = 0.0
        # Do not compare potentials across different objectives (e.g. after a
        # checkpoint or coin), and do not reward the position reached on a
        # death-triggering frame.
        if not objective_changed and not info.get("death_triggered", False):
            progress = float(old_distance - new_distance)
            # Do not clip exact potential differences. Clipping only one side of
            # a multi-step cycle can create positive-return reward loops.
            if self.clip_progress is not None:
                progress = float(np.clip(progress, -self.clip_progress, self.clip_progress))
            self._best_distance = min(self._best_distance, new_distance)
        else:
            self._segment_start_distance = new_distance
            self._best_distance = new_distance

        reward = self.step_penalty + self.progress_scale * progress
        if info.get("death_triggered", False):
            reward += self.death_penalty
        if info.get("checkpoint_reached"):
            reward += self.checkpoint_bonus
        if info.get("coins_collected_this_step", 0):
            reward += self.coin_bonus * float(info["coins_collected_this_step"])
        if info.get("level_cleared_this_step", False):
            reward += self.clear_bonus

        self._objective = new_objective
        self._distance = new_distance
        max_progress = max(0.0, self._segment_start_distance - self._best_distance)
        success = bool(info.get("level_cleared_this_step", False))
        info.update({
            "path_objective": new_objective.kind,
            "path_distance": new_distance,
            "path_progress": progress,
            "max_path_progress": max_progress,
            "is_success": success,
            "shaped_reward": float(reward),
        })
        return obs, float(reward), terminated, truncated, info
