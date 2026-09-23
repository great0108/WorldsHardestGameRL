from __future__ import annotations

from typing import Any
import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .core import WorldsHardestGameCore
from .game_data import COIN_COUNTS, STAGE_SIZE, GameData
from .render import Renderer

MAX_ENEMIES = 320
MAX_COINS = 67
# 11 scalars + enemy (dx,dy,mask) + coin (dx,dy,mask)
STATE_SIZE = 11 + MAX_ENEMIES * 3 + MAX_COINS * 3


class WorldsHardestGameEnv(gym.Env):
    """Gymnasium wrapper around the reconstructed 30-level WHG core.

    Parameters
    ----------
    level:
        1..30 for a fixed level, or None for the full campaign.
    observation_mode:
        ``"state"`` for a compact vector or ``"pixels"`` for 550x400 RGB.
    reward_mode:
        ``"sparse"`` or ``"dense"``. Rewards are RL conveniences and are not
        part of the original Flash game physics.
    death_ends_episode:
        If true, the frame that starts a death terminates the episode. Useful
        for per-level training. If false, the original fade/respawn plays out.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        level: int | None = 1,
        observation_mode: str = "state",
        reward_mode: str = "sparse",
        death_ends_episode: bool = False,
        render_mode: str | None = None,
        max_steps: int | None = 30_000,
        swf_path: str | None = None,
        pixel_size: tuple[int, int] | None = None,
        pixel_resize_method: str = "box",
    ):
        super().__init__()
        if level is not None and not 1 <= int(level) <= 30:
            raise ValueError("level must be 1..30 or None")
        if observation_mode not in {"state", "pixels"}:
            raise ValueError("observation_mode must be 'state' or 'pixels'")
        if reward_mode not in {"sparse", "dense", "zero"}:
            raise ValueError("reward_mode must be 'sparse', 'dense', or 'zero'")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")

        self.fixed_level = int(level) if level is not None else None
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.death_ends_episode = bool(death_ends_episode)
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.pixel_size = tuple(map(int, pixel_size)) if pixel_size is not None else None
        self.pixel_resize_method = str(pixel_resize_method)
        if self.pixel_size is not None and (self.pixel_size[0] < 1 or self.pixel_size[1] < 1):
            raise ValueError("pixel_size values must be positive")
        if self.pixel_resize_method not in {"nearest", "bilinear", "box"}:
            raise ValueError("pixel_resize_method must be nearest, bilinear, or box")

        self.data = GameData(swf_path) if swf_path else GameData()
        self.core = WorldsHardestGameCore(
            self.data,
            level=self.fixed_level or 1,
            campaign=self.fixed_level is None,
        )
        self.renderer = Renderer(self.data)
        self.action_space = spaces.Discrete(9)
        if observation_mode == "state":
            self.observation_space = spaces.Box(
                low=-2.0, high=2.0, shape=(STATE_SIZE,), dtype=np.float32
            )
        else:
            if self.pixel_size is None:
                pixel_w, pixel_h = STAGE_SIZE
            else:
                pixel_w, pixel_h = self.pixel_size
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(pixel_h, pixel_w, 3), dtype=np.uint8
            )
        self._steps = 0
        self._screen = None
        self._clock = None

    def _target_center(self) -> tuple[float, float]:
        d = self.core.defn
        # For multi-checkpoint levels, the next unvisited checkpoint is the
        # useful navigation target; after that, use the actual goal.
        if self.core.level == 5:
            if self.core.current_check == "check1":
                return self.data.instance_center(d.checks["check2"])
            if self.core.current_check == "check2":
                return self.data.instance_center(d.checks["check3"])
        elif self.core.level in {6,9,12,27,28} and self.core.current_check == "check1":
            return self.data.instance_center(d.checks["check2"])
        return self.data.instance_center(d.checks[d.goal_name])

    def _distance_to_target(self) -> float:
        tx, ty = self._target_center()
        return math.hypot(tx-self.core.player_x, ty-self.core.player_y)

    def _state_obs(self) -> np.ndarray:
        c = self.core
        d = c.defn
        tx, ty = self._target_center()
        goal = self.data.instance_center(d.checks[d.goal_name])
        check_idx = int(c.current_check.replace("check", ""))
        required = COIN_COUNTS[c.level-1]
        values: list[float] = [
            c.level / 30.0,
            c.player_x / 550.0,
            c.player_y / 400.0,
            c.alpha / 100.0,
            1.0 if c.move_ready else 0.0,
            check_idx / 4.0,
            c.current_coins / MAX_COINS,
            required / MAX_COINS,
            (tx-c.player_x) / 550.0,
            (ty-c.player_y) / 400.0,
            math.hypot(goal[0]-c.player_x, goal[1]-c.player_y) / math.hypot(550,400),
        ]

        enemies = sorted(
            c.enemy_centers(),
            key=lambda p: (p[0]-c.player_x)**2 + (p[1]-c.player_y)**2,
        )[:MAX_ENEMIES]
        for x,y in enemies:
            values.extend(((x-c.player_x)/550.0, (y-c.player_y)/400.0, 1.0))
        values.extend((0.0,0.0,0.0) * (MAX_ENEMIES-len(enemies)))

        visible_coins = [coin for coin,taken in zip(d.coins,c.collected) if not taken]
        visible_coins.sort(key=lambda coin: (
            (self.data.instance_center(coin)[0]-c.player_x)**2 +
            (self.data.instance_center(coin)[1]-c.player_y)**2
        ))
        visible_coins = visible_coins[:MAX_COINS]
        for coin in visible_coins:
            x,y=self.data.instance_center(coin)
            values.extend(((x-c.player_x)/550.0, (y-c.player_y)/400.0, 1.0))
        values.extend((0.0,0.0,0.0) * (MAX_COINS-len(visible_coins)))
        return np.asarray(values, dtype=np.float32)

    def _obs(self):
        if self.observation_mode == "pixels":
            if self.pixel_size is None:
                return self.renderer.rgb(self.core)
            return self.renderer.rgb_legacy_resized(
                self.core, self.pixel_size, method=self.pixel_resize_method
            )
        return self._state_obs()

    def _info(self, step_info=None) -> dict[str, Any]:
        out = self.core.clone_state()
        out.update({
            "required_coins": COIN_COUNTS[self.core.level-1],
            "fixed_level": self.fixed_level,
            "flash_frame_seconds": 1/30,
        })
        if step_info is not None:
            out.update({
                "level_cleared_this_step": step_info.level_cleared,
                "game_cleared_this_step": step_info.game_cleared,
                "level_advanced_this_step": step_info.level_advanced,
                "death_triggered": step_info.death_triggered,
                "death_completed": step_info.death_completed,
                "checkpoint_reached": step_info.checkpoint_reached,
                "coins_collected_this_step": step_info.coins_collected,
                "coins_reset_this_step": step_info.coins_reset,
            })
        return out

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        level = options.get("level", self.fixed_level or 1)
        if self.fixed_level is not None:
            level = self.fixed_level
        self.core.campaign = self.fixed_level is None
        self.core.reset(level=int(level), deaths=0)
        self._steps = 0
        obs = self._obs()
        if self.render_mode == "human":
            self.render()
        return obs, self._info()

    def step(self, action):
        old_level = self.core.level
        old_dist = self._distance_to_target()
        info = self.core.step_frame(int(action))
        self._steps += 1

        if self.reward_mode == "zero":
            reward = 0.0
        else:
            reward = -0.001
            reward += 0.05 * info.coins_collected
            if info.checkpoint_reached:
                reward += 0.2
            if info.death_triggered:
                reward -= 1.0
            if info.level_cleared:
                reward += 2.0
            if info.game_cleared:
                reward += 20.0
            if self.reward_mode == "dense" and self.core.level == old_level and self.core.move_ready:
                reward += (old_dist - self._distance_to_target()) / 550.0

        if self.fixed_level is None:
            terminated = bool(info.game_cleared)
        else:
            terminated = bool(info.level_cleared)
        if self.death_ends_episode and info.death_triggered:
            terminated = True
        truncated = bool(self.max_steps is not None and self._steps >= self.max_steps)

        obs = self._obs()
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, self._info(info)

    def render(self):
        frame = self.renderer.rgb(self.core)
        if self.render_mode == "rgb_array" or self.render_mode is None:
            return frame
        try:
            import pygame
        except ImportError as e:
            raise RuntimeError("human rendering requires pygame: pip install pygame") from e
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode(STAGE_SIZE)
            pygame.display.set_caption("The World's Hardest Game — SWF Gym reconstruction")
            self._clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1,0,2)))
        self._screen.blit(surf, (0,0))
        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(30)
        return None

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            self._screen = None
