from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Hashable

import numpy as np

from .core import ACTION_TO_VECTOR, WorldsHardestGameCore
from .game_data import (
    COIN_COUNTS, SOURCE_SHA1,
    FOUR_CHECK_LEVELS,
    PLAYER_HALF_H,
    PLAYER_HALF_W,
    PLAYER_SPEED,
    THREE_CHECK_LEVELS,
    Instance, STAGE_SIZE,
)


@dataclass(frozen=True, slots=True)
class NavigationObjective:
    """Current geometry-only navigation objective used for reward shaping."""

    key: Hashable
    instance: Instance
    kind: str


class ShortestPathOracle:
    """Exact geometry-only shortest-path oracle for reward shaping.

    Unlike the old raster walkability approximation, this oracle's state graph
    is made from positions that the authored player physics can *actually*
    reach. Every edge applies the exact 3 px action followed by the exact Flash
    wall-correction order. This includes transient wall-overlap positions that
    are reachable for one or more frames and were previously misclassified as
    non-walkable.

    Moving enemies are deliberately ignored so the pixel policy still has to
    learn *when* it is safe to move. Permanently stationary enemy shapes are
    treated as lethal static obstacles using the same five collision probes as
    the game, so shortest paths never route through them.
    """

    def __init__(self, core: WorldsHardestGameCore, cache_path: str | Path | None = None):
        self.core = core
        self.data = core.data
        self.level = core.level

        if cache_path is None:
            arrays = build_spatial_oracle_arrays(core)
        else:
            with np.load(Path(cache_path), allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}

        if int(arrays["cache_version"]) != SPATIAL_ORACLE_CACHE_VERSION:
            raise ValueError("spatial oracle cache version mismatch")
        if int(arrays["level"]) != self.level:
            raise ValueError("spatial oracle cache level mismatch")
        if str(arrays["source_sha1"].item()) != core.data.sha1:
            raise ValueError("spatial oracle cache SWF hash mismatch")

        self.positions = np.asarray(arrays["positions"], dtype=np.float32)
        self.transitions = np.asarray(arrays["transitions"])
        self.static_hazard = np.asarray(arrays["static_hazard"], dtype=np.bool_)
        self.static_enemy_count = int(arrays["static_enemy_count"])
        self._node_for_xy = {
            (round(float(x), 6), round(float(y), 6)): i
            for i, (x, y) in enumerate(self.positions)
        }

        self.predecessors: list[list[int]] = [[] for _ in range(len(self.positions))]
        for src, row in enumerate(self.transitions):
            if self.static_hazard[src]:
                continue
            for dst_raw in row:
                dst = int(dst_raw)
                # Entering a stationary enemy would trigger death after wall
                # correction, so there is no alive navigation edge into it.
                if self.static_hazard[dst]:
                    continue
                self.predecessors[dst].append(src)

        # objective.key -> (distance field, finite dead-end sentinel)
        self._fields: dict[Hashable, tuple[np.ndarray, float]] = {}

    def _node(self, x: float, y: float) -> int | None:
        return self._node_for_xy.get((round(float(x), 6), round(float(y), 6)))

    def _target_nodes(self, inst: Instance) -> np.ndarray:
        tx0, tx1, ty0, ty1 = self.data.instance_bounds(inst)
        xs = self.positions[:, 0]
        ys = self.positions[:, 1]
        overlap = (
            (xs + PLAYER_HALF_W >= tx0)
            & (xs - PLAYER_HALF_W <= tx1)
            & (ys + PLAYER_HALF_H >= ty0)
            & (ys - PLAYER_HALF_H <= ty1)
            & (~self.static_hazard)
        )
        nodes = np.flatnonzero(overlap)
        if len(nodes):
            return nodes

        # Defensive fallback for a very thin/oddly transformed objective. Pick
        # the nearest *safe graph node*. This never changes graph connectivity.
        safe = np.flatnonzero(~self.static_hazard)
        if not len(safe):
            return safe
        cx, cy = self.data.instance_center(inst)
        pts = self.positions[safe]
        d2 = (pts[:, 0] - float(cx)) ** 2 + (pts[:, 1] - float(cy)) ** 2
        return np.asarray([safe[int(np.argmin(d2))]], dtype=np.int64)

    def _field_for(self, objective: NavigationObjective) -> tuple[np.ndarray, float]:
        cached = self._fields.get(objective.key)
        if cached is not None:
            return cached

        n = len(self.positions)
        dist = np.full(n, np.inf, dtype=np.float32)
        q: deque[int] = deque()

        for node_raw in self._target_nodes(objective.instance):
            node = int(node_raw)
            dist[node] = 0.0
            q.append(node)

        # Reverse BFS on the *actual directed physics graph*. Wall correction
        # can make edges asymmetric, so an 8-neighbour lattice is not exact.
        while q:
            node = q.popleft()
            nd = float(dist[node] + 1.0)
            for prev in self.predecessors[node]:
                if dist[prev] <= nd:
                    continue
                dist[prev] = nd
                q.append(prev)

        finite = dist[np.isfinite(dist)]
        # No Euclidean fallback: an unreachable/unknown state gets a finite
        # sentinel outside the reachable distance range. This avoids mixing two
        # incompatible distance metrics near walls.
        dead_end_distance = float(finite.max() + 1.0) if finite.size else float(n + 1)
        cached = (dist, dead_end_distance)
        self._fields[objective.key] = cached
        return cached

    def distance(self, x: float, y: float, objective: NavigationObjective) -> float:
        dist, dead_end_distance = self._field_for(objective)
        node = self._node(x, y)
        if node is None or self.static_hazard[node]:
            return dead_end_distance
        value = float(dist[node])
        return value if math.isfinite(value) else dead_end_distance

def current_navigation_objective(core: WorldsHardestGameCore) -> NavigationObjective:
    """Return the next useful static objective for geometry-based shaping.

    Intermediate checkpoints take precedence. After checkpoint prerequisites
    are satisfied, the closest remaining required coin is targeted; finally
    the authored goal/check area is targeted.
    """

    d = core.defn
    n = core.level

    if n in FOUR_CHECK_LEVELS:
        if core.current_check == "check1":
            return NavigationObjective((n, "check2"), d.checks["check2"], "checkpoint")
        if core.current_check == "check2":
            return NavigationObjective((n, "check3"), d.checks["check3"], "checkpoint")
    elif n in THREE_CHECK_LEVELS and core.current_check == "check1":
        return NavigationObjective((n, "check2"), d.checks["check2"], "checkpoint")

    required = COIN_COUNTS[n-1]
    if core.current_coins < required:
        remaining = [(i, coin) for i, (coin, taken) in enumerate(zip(d.coins, core.collected)) if not taken]
        if remaining:
            px, py = core.player_x, core.player_y
            i, coin = min(
                remaining,
                key=lambda item: (
                    (core.data.instance_center(item[1])[0]-px)**2
                    + (core.data.instance_center(item[1])[1]-py)**2
                ),
            )
            return NavigationObjective((n, "coin", i), coin, "coin")

    return NavigationObjective((n, d.goal_name), d.checks[d.goal_name], "goal")


SPATIAL_ORACLE_CACHE_VERSION = 1


def _matrix_equal(a, b, eps: float = 1e-9) -> bool:
    return (
        abs(a.sx-b.sx) <= eps and abs(a.sy-b.sy) <= eps
        and abs(a.r0-b.r0) <= eps and abs(a.r1-b.r1) <= eps
        and abs(a.tx-b.tx) <= eps and abs(a.ty-b.ty) <= eps
    )


def _find_static_enemy_leaves(core: WorldsHardestGameCore):
    """Return enemy shape leaves whose full authored transform never changes."""
    swf = core.data.swf
    out = []

    def visit(sid, matrix):
        if sid in swf.shape_records:
            out.append((swf.shape(sid), matrix))
            return
        spr = swf.sprite(sid)
        if not spr.frames:
            return
        common_depths = set(spr.frames[0])
        for state in spr.frames[1:]:
            common_depths.intersection_update(state)
        for depth in sorted(common_depths):
            first = spr.frames[0][depth]
            constant = True
            for state in spr.frames[1:]:
                obj = state[depth]
                if obj.char != first.char or not _matrix_equal(obj.matrix, first.matrix):
                    constant = False
                    break
            if constant:
                visit(first.char, matrix.then(first.matrix))

    visit(core.defn.enemies.symbol, core.defn.enemies.matrix)
    return tuple(out)


def _static_enemy_collision_at(
    core: WorldsHardestGameCore,
    static_leaves,
    x: float,
    y: float,
) -> bool:
    """Exact five-probe enemy collision against only stationary enemy leaves."""
    if not static_leaves:
        return False
    probes = (
        (x, y),
        (x + PLAYER_HALF_W, y),
        (x - PLAYER_HALF_W, y),
        (x, y - PLAYER_HALF_H),
        (x, y + PLAYER_HALF_H),
    )
    return any(core._leaf_hit(static_leaves, px, py) for px, py in probes)


def build_spatial_oracle_arrays(core: WorldsHardestGameCore) -> dict[str, np.ndarray]:
    """Build an exact, geometry-only player-position graph for one level.

    Graph discovery follows the exact authored wall correction but ignores all
    enemy motion. It starts from every authored checkpoint spawn so checkpoint
    respawns are represented even when their 3 px lattice offset differs from
    check1. Stationary enemies are then marked as lethal nodes and excluded from
    navigation edges/paths; moving enemies remain absent from the oracle.
    """
    # Seed every checkpoint registration position, not just check1. This makes
    # all legitimate respawn lattices available to the runtime oracle.
    starts: list[tuple[float, float]] = []
    for inst in core.defn.checks.values():
        xy = (round(float(inst.matrix.tx), 6), round(float(inst.matrix.ty), 6))
        if xy not in starts:
            starts.append(xy)

    queue: deque[tuple[float, float]] = deque(starts)
    node_for_xy: dict[tuple[float, float], int] = {
        xy: i for i, xy in enumerate(starts)
    }
    positions: list[tuple[float, float]] = list(starts)
    transition_rows: list[list[int] | None] = [None] * len(starts)

    # Exact vector hit tests are relatively expensive. The same wall probe
    # coordinates recur across many neighbouring states/actions, so memoize
    # them without changing collision semantics.
    wall_hit_cache: dict[tuple[float, float], bool] = {}

    def wall_hit_cached(px: float, py: float) -> bool:
        key = (round(float(px), 6), round(float(py), 6))
        value = wall_hit_cache.get(key)
        if value is None:
            value = bool(core.wall_hit(px, py))
            wall_hit_cache[key] = value
        return value

    def transition(x: float, y: float, action: int) -> tuple[float, float]:
        dx, dy = ACTION_TO_VECTOR[int(action)]
        x = float(x) + float(dx) * PLAYER_SPEED
        y = float(y) + float(dy) * PLAYER_SPEED
        if wall_hit_cached(x + PLAYER_HALF_W - PLAYER_SPEED, y):
            x -= PLAYER_SPEED
        if wall_hit_cached(x - PLAYER_HALF_W + PLAYER_SPEED, y):
            x += PLAYER_SPEED
        if wall_hit_cached(x, y - PLAYER_HALF_H + PLAYER_SPEED):
            y += PLAYER_SPEED
        if wall_hit_cached(x, y + PLAYER_HALF_H - PLAYER_SPEED):
            y -= PLAYER_SPEED
        return round(x, 6), round(y, 6)

    while queue:
        x, y = queue.popleft()
        src = node_for_xy[(x, y)]
        row: list[int] = []
        for action in range(9):
            nxt = transition(x, y, action)
            if not (
                -20.0 <= nxt[0] <= STAGE_SIZE[0] + 20.0
                and -20.0 <= nxt[1] <= STAGE_SIZE[1] + 20.0
            ):
                raise RuntimeError(f"spatial wall transition escaped stage: {(x, y)} -> {nxt}")
            node = node_for_xy.get(nxt)
            if node is None:
                node = len(positions)
                if node >= 100_000:
                    raise RuntimeError("spatial geometry graph grew unexpectedly large")
                node_for_xy[nxt] = node
                positions.append(nxt)
                transition_rows.append(None)
                queue.append(nxt)
            row.append(node)
        transition_rows[src] = row

    if any(row is None for row in transition_rows):
        raise RuntimeError("spatial geometry graph contains an unexpanded node")

    pos = np.asarray(positions, dtype=np.float32)
    transitions = np.asarray(
        transition_rows,
        dtype=np.int16 if len(pos) < 32767 else np.int32,
    )

    static_leaves = _find_static_enemy_leaves(core)
    static_hazard = np.fromiter(
        (
            _static_enemy_collision_at(core, static_leaves, float(x), float(y))
            for x, y in positions
        ),
        dtype=np.bool_,
        count=len(positions),
    )

    return {
        "cache_version": np.asarray(SPATIAL_ORACLE_CACHE_VERSION, dtype=np.int32),
        "level": np.asarray(core.level, dtype=np.int16),
        "source_sha1": np.asarray(SOURCE_SHA1),
        "positions": pos,
        "transitions": transitions,
        "static_hazard": static_hazard,
        "static_enemy_count": np.asarray(len(static_leaves), dtype=np.int32),
    }


def ensure_spatial_oracle_cache(
    path: str | Path,
    *,
    level: int,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Build the exact spatial graph once in the parent process and cache it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def valid_existing() -> bool:
        if force or not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                return (
                    int(data["cache_version"]) == SPATIAL_ORACLE_CACHE_VERSION
                    and int(data["level"]) == int(level)
                    and str(data["source_sha1"].item()) == SOURCE_SHA1
                    and "positions" in data
                    and "transitions" in data
                    and "static_hazard" in data
                )
        except Exception:
            return False

    if valid_existing():
        if verbose:
            print(f"Spatial oracle cache: {path} (reused)")
        return path

    if verbose:
        print(f"Building exact spatial oracle for level {level} (one-time cache)...")
    core = WorldsHardestGameCore(level=level, campaign=False)
    arrays = build_spatial_oracle_arrays(core)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)

    if verbose:
        print(
            "Spatial oracle ready: "
            f"positions={len(arrays['positions'])}, "
            f"static_enemy_leaves={int(arrays['static_enemy_count'])}, "
            f"static_hazard_nodes={int(np.count_nonzero(arrays['static_hazard']))}, "
            f"cache={path}"
        )
    return path
