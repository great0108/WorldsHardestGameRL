from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from .swf import Matrix, SWF, DisplayObject

SOURCE_SHA1 = "cd9a441052fcadc2522ad09e48156f5c604caa16"
SOURCE_MD5 = "fba226c3a20aa421d4f8bd3c9b26d5ea"
STAGE_SIZE = (550, 400)
FPS = 30
PLAYER_SPEED = 3.0
PLAYER_HALF_W = 9.0
PLAYER_HALF_H = 9.0
COIN_SYMBOL = 115
# Extracted from root frame 61 ActionScript: coins[currentLevel - 1]
COIN_COUNTS = (0,1,1,3,0,4,4,3,1,0,2,1,0,0,0,4,0,67,0,7,3,0,36,0,0,4,1,0,2,4)
THREE_CHECK_LEVELS = frozenset({6,9,12,27,28})
FOUR_CHECK_LEVELS = frozenset({5})


@dataclass(frozen=True, slots=True)
class Instance:
    symbol: int
    matrix: Matrix
    name: str | None = None
    depth: int = 0


@dataclass(frozen=True, slots=True)
class LevelDefinition:
    number: int
    root_frame: int
    background: Instance
    walls: Instance
    enemies: Instance
    checks: dict[str, Instance]
    coins: tuple[Instance, ...]

    @property
    def start_name(self) -> str:
        return "check1"

    @property
    def goal_name(self) -> str:
        if self.number in FOUR_CHECK_LEVELS:
            return "check4"
        if self.number in THREE_CHECK_LEVELS:
            return "check3"
        return "check2"


class GameData:
    def __init__(self, swf_path: str | Path | None = None, verify_hash: bool = True):
        if swf_path is None:
            swf_path = Path(__file__).with_name("assets") / "WHGOriginal.swf"
        self.path = Path(swf_path)
        raw = self.path.read_bytes()
        self.sha1 = hashlib.sha1(raw).hexdigest()
        self.md5 = hashlib.md5(raw).hexdigest()
        if verify_hash and self.sha1 != SOURCE_SHA1:
            raise ValueError(
                f"This build targets WHGOriginal.swf SHA-1 {SOURCE_SHA1}, "
                f"but got {self.sha1}. Pass verify_hash=False only if you intentionally want to inspect another build."
            )
        self.swf = SWF(self.path)
        self.levels = tuple(self._extract_level(i) for i in range(1,31))
        self._validate()

    @staticmethod
    def _inst(o: DisplayObject) -> Instance:
        return Instance(o.char,o.matrix,o.name,o.depth)

    def _extract_level(self, level: int) -> LevelDefinition:
        frame = 71 + (level-1)*4
        places = self.swf.root_places_at(frame)
        by_name = {o.name:o for o in places if o.name}
        if "walls" not in by_name or "enemies" not in by_name:
            raise ValueError(f"level {level}: missing named gameplay clips")
        checks = {name:self._inst(o) for name,o in by_name.items() if name and name.startswith("check")}
        backgrounds = [o for o in places if o.depth == 1 and not o.name]
        if not backgrounds:
            raise ValueError(f"level {level}: missing background")
        coin_objs = [o for o in places if o.char == COIN_SYMBOL]
        # Level 9 contains one extra coin instance at depth 8, underneath the
        # walls clip at depth 10. It is invisible during normal play and is not
        # part of the scripted one-coin requirement. Exclude it entirely from
        # the reconstructed Gym environment so observations, collision/collection
        # logic, and rendering contain only the visible gameplay coin.
        if level == 9:
            coin_objs = [o for o in coin_objs if o.depth > by_name["walls"].depth]
        coins = tuple(self._inst(o) for o in coin_objs)
        return LevelDefinition(
            number=level,
            root_frame=frame,
            background=self._inst(backgrounds[0]),
            walls=self._inst(by_name["walls"]),
            enemies=self._inst(by_name["enemies"]),
            checks=checks,
            coins=coins,
        )

    def _validate(self) -> None:
        if self.swf.stage_rect != (0.0,550.0,0.0,400.0):
            raise ValueError(f"unexpected stage {self.swf.stage_rect}")
        if abs(self.swf.fps-30.0)>1e-9:
            raise ValueError(f"unexpected fps {self.swf.fps}")
        for level, expected in zip(self.levels, COIN_COUNTS):
            if len(level.coins) < expected:
                raise ValueError(f"level {level.number}: SWF has only {len(level.coins)} coin clips, script requires {expected}")
            if "check1" not in level.checks or level.goal_name not in level.checks:
                raise ValueError(f"level {level.number}: missing start/goal check")

    def level(self, n: int) -> LevelDefinition:
        if not 1 <= n <= 30:
            raise ValueError("level must be 1..30")
        return self.levels[n-1]

    def instance_bounds(self, inst: Instance) -> tuple[float,float,float,float]:
        b=self.swf.symbol_bounds(inst.symbol)
        pts=[inst.matrix.apply(x,y) for x in (b[0],b[1]) for y in (b[2],b[3])]
        return min(x for x,_ in pts),max(x for x,_ in pts),min(y for _,y in pts),max(y for _,y in pts)

    def instance_center(self, inst: Instance) -> tuple[float,float]:
        x0,x1,y0,y1=self.instance_bounds(inst)
        return (x0+x1)/2,(y0+y1)/2
