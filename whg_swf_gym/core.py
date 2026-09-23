from __future__ import annotations

from dataclasses import dataclass
from .game_data import (
    PLAYER_HALF_H, PLAYER_HALF_W, PLAYER_SPEED,
    FOUR_CHECK_LEVELS, THREE_CHECK_LEVELS, COIN_COUNTS, GameData, Instance,
)

ACTION_TO_VECTOR = {
    0:(0,0), 1:(0,-1), 2:(0,1), 3:(-1,0), 4:(1,0),
    5:(-1,-1), 6:(1,-1), 7:(-1,1), 8:(1,1),
}


@dataclass(slots=True)
class StepInfo:
    level: int
    level_cleared: bool = False
    game_cleared: bool = False
    level_advanced: bool = False
    death_triggered: bool = False
    death_completed: bool = False
    wall_corrected: bool = False
    checkpoint_reached: str | None = None
    coins_collected: int = 0
    coins_reset: int = 0


class WorldsHardestGameCore:
    """Physics/gameplay core reconstructed from WHGOriginal.swf.

    One step is one Flash frame (30 Hz).  The keyboard movement, wall point
    tests, enemy five-point tests, death alpha fade, checkpoint respawn, coin
    reset-on-death behavior, and level-specific checkpoint scripts follow the
    ActionScript embedded in the supplied SWF.
    """

    def __init__(self, data: GameData | None = None, level: int = 1, campaign: bool = True):
        self.data = data or GameData()
        self.campaign = bool(campaign)
        self.deaths = 0
        self.total_frames = 0
        self.game_cleared = False
        self.set_level(level, preserve_deaths=True)

    def set_level(self, level: int, *, preserve_deaths: bool = True) -> None:
        if not preserve_deaths:
            self.deaths = 0
        self.level = int(level)
        self.defn = self.data.level(self.level)
        self.current_check = "check1"
        # onClipEvent(load): player._x/_y = currentCheck._x/_y. _x/_y are the
        # instance registration-point coordinates, i.e. the root matrix tx/ty.
        start = self.defn.checks["check1"].matrix
        self.player_x = float(start.tx)
        self.player_y = float(start.ty)
        self.alpha = 100.0
        self.move_ready = True
        self.enemy_frame = 0
        self.collected = [False] * len(self.defn.coins)
        self.level_frames = 0
        self.level_cleared = False
        self._wall_leaves = tuple(self.data.swf.flatten(
            self.defn.walls.symbol, self.defn.walls.matrix, 0
        ))

    def reset(self, *, level: int | None = None, deaths: int = 0) -> None:
        self.deaths = int(deaths)
        self.total_frames = 0
        self.game_cleared = False
        self.set_level(level or self.level, preserve_deaths=True)

    @staticmethod
    def _aabb_overlap(a, b) -> bool:
        ax0,ax1,ay0,ay1=a; bx0,bx1,by0,by1=b
        return ax1 >= bx0 and ax0 <= bx1 and ay1 >= by0 and ay0 <= by1

    @property
    def player_bounds(self) -> tuple[float,float,float,float]:
        return (
            self.player_x-PLAYER_HALF_W, self.player_x+PLAYER_HALF_W,
            self.player_y-PLAYER_HALF_H, self.player_y+PLAYER_HALF_H,
        )

    def _instance_overlap_player(self, inst: Instance) -> bool:
        return self._aabb_overlap(self.player_bounds, self.data.instance_bounds(inst))

    @staticmethod
    def _leaf_hit(leaves, x: float, y: float) -> bool:
        for shape, matrix in leaves:
            lx,ly = matrix.inverse_apply(x,y)
            if shape.hit(lx,ly):
                return True
        return False

    def wall_hit(self, x: float, y: float) -> bool:
        return self._leaf_hit(self._wall_leaves,x,y)

    def enemy_leaves(self):
        return tuple(self.data.swf.flatten(
            self.defn.enemies.symbol, self.defn.enemies.matrix, self.enemy_frame
        ))

    def enemy_hit(self, x: float, y: float) -> bool:
        return self._leaf_hit(self.enemy_leaves(),x,y)

    def enemy_centers(self) -> tuple[tuple[float,float], ...]:
        # Every leaf in the WHG enemies clips is an enemy vector shape centered
        # at its symbol origin.  This is useful for state observations.
        return tuple(m.apply(0.0,0.0) for _,m in self.enemy_leaves())

    def _coin_events(self, info: StepInfo) -> None:
        # Coin onClipEvent(enterFrame): collect only at alpha==100; restore all
        # collected coins when player alpha<2 during the death fade.
        if self.alpha < 2:
            reset = sum(self.collected)
            if reset:
                self.collected = [False] * len(self.collected)
                info.coins_reset = reset
            return
        if self.alpha != 100:
            return
        for i,(coin,taken) in enumerate(zip(self.defn.coins,self.collected)):
            if not taken and self._instance_overlap_player(coin):
                self.collected[i] = True
                info.coins_collected += 1

    @property
    def current_coins(self) -> int:
        return sum(self.collected)

    def _check_logic_and_goal(self, info: StepInfo) -> bool:
        n=self.level
        if n in FOUR_CHECK_LEVELS:
            if self.current_check == "check1" and self._instance_overlap_player(self.defn.checks["check2"]):
                self.current_check="check2"; info.checkpoint_reached="check2"
            if self.current_check in ("check1","check2") and self._instance_overlap_player(self.defn.checks["check3"]):
                self.current_check="check3"; info.checkpoint_reached="check3"
        elif n in THREE_CHECK_LEVELS:
            if self.current_check == "check1" and self._instance_overlap_player(self.defn.checks["check2"]):
                self.current_check="check2"; info.checkpoint_reached="check2"

        goal=self.defn.checks[self.defn.goal_name]
        if self._instance_overlap_player(goal) and self.current_coins == COIN_COUNTS[self.level-1]:
            info.level_cleared=True
            self.level_cleared=True
            if n==30:
                info.game_cleared=True
                self.game_cleared=True
            elif self.campaign:
                self.set_level(n+1,preserve_deaths=True)
                info.level_advanced=True
                info.level=n+1
            return True
        return False

    def _apply_action(self, action: int) -> None:
        dx,dy=ACTION_TO_VECTOR[action]
        self.player_x += dx*PLAYER_SPEED
        self.player_y += dy*PLAYER_SPEED

    def _correct_walls(self) -> bool:
        corrected=False
        # Exact order and probe coordinates from the shared player AS2 script.
        if self.wall_hit(self.player_x + PLAYER_HALF_W - PLAYER_SPEED,self.player_y):
            self.player_x -= PLAYER_SPEED; corrected=True
        if self.wall_hit(self.player_x - PLAYER_HALF_W + PLAYER_SPEED,self.player_y):
            self.player_x += PLAYER_SPEED; corrected=True
        if self.wall_hit(self.player_x,self.player_y - PLAYER_HALF_H + PLAYER_SPEED):
            self.player_y += PLAYER_SPEED; corrected=True
        if self.wall_hit(self.player_x,self.player_y + PLAYER_HALF_H - PLAYER_SPEED):
            self.player_y -= PLAYER_SPEED; corrected=True
        return corrected

    def _enemy_collision(self) -> bool:
        leaves=self.enemy_leaves()
        points=(
            (self.player_x,self.player_y),
            (self.player_x+PLAYER_HALF_W,self.player_y),
            (self.player_x-PLAYER_HALF_W,self.player_y),
            (self.player_x,self.player_y-PLAYER_HALF_H),
            (self.player_x,self.player_y+PLAYER_HALF_H),
        )
        return any(self._leaf_hit(leaves,x,y) for x,y in points)

    def step_frame(self, action: int) -> StepInfo:
        if action not in ACTION_TO_VECTOR:
            raise ValueError(f"action must be 0..8, got {action!r}")
        info=StepInfo(level=self.level)

        # Flash's enterFrame handler runs after the playhead has entered the
        # frame that will be rendered. Keep the enemy timeline on that same
        # frame for both collision tests and rendering. On the first gameplay
        # tick the newly-instantiated enemy clips are still on authored frame 1
        # (our index 0); subsequent ticks advance before the player's handler.
        if self.level_frames > 0:
            self.enemy_frame += 1

        # Coin clips live at lower display depths than player and evaluate the
        # player's current (pre-keyboard) position.
        self._coin_events(info)

        # Player enterFrame starts with checkpoints/goal before death/movement.
        if self._check_logic_and_goal(info):
            self.total_frames += 1
            if not info.level_advanced:
                self.level_frames += 1
            return info

        if not self.move_ready:
            if self.alpha > 0:
                self.alpha -= 4
            else:
                self.deaths += 1
                self.alpha = 100.0
                cp=self.defn.checks[self.current_check].matrix
                self.player_x=float(cp.tx); self.player_y=float(cp.ty)
                self.move_ready=True
                info.death_completed=True

        if self.move_ready:
            self._apply_action(action)

        info.wall_corrected=self._correct_walls()

        if self._enemy_collision() and self.alpha == 100:
            self.move_ready=False
            info.death_triggered=True

        # Movie clips keep animating while the player fades; their playhead
        # advancement happens at the start of the next Flash frame above.
        self.total_frames += 1
        self.level_frames += 1
        return info

    def clone_state(self) -> dict:
        return {
            "level":self.level,"player_x":self.player_x,"player_y":self.player_y,
            "alpha":self.alpha,"move_ready":self.move_ready,"deaths":self.deaths,
            "enemy_frame":self.enemy_frame,"current_check":self.current_check,
            "coins":self.current_coins,"coin_mask":tuple(self.collected),
            "level_cleared":self.level_cleared,"game_cleared":self.game_cleared,
            "total_frames":self.total_frames,"level_frames":self.level_frames,
        }
