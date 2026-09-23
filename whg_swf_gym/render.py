from __future__ import annotations

from PIL import Image, ImageDraw
import numpy as np

from .game_data import STAGE_SIZE, COIN_COUNTS, GameData, Instance
from .core import WorldsHardestGameCore
from .swf import Matrix, Shape


class Renderer:
    """WHG vector renderer with resolution-aware static-layer caching.

    Physics always stays in the source 550x400 Flash coordinate system.  The
    renderer can either draw at the source resolution or directly into a
    smaller framebuffer (for example 110x80 for pixel RL).  Static level art is
    cached separately for every ``(level, width, height)`` tuple, so training
    frames only copy a pre-rendered background and paint moving objects.
    """

    def __init__(self, data: GameData):
        self.data = data
        self._static_cache: dict[tuple[int, int, int], Image.Image] = {}
        self._hud_cache: dict[tuple[int, int, int], Image.Image] = {}
        # Final opaque HUD strips are cached by the values that actually change.
        # This removes per-frame font rasterization while keeping exact pixels.
        self._hud_top_cache: dict[tuple[int, int, int, int, int], Image.Image] = {}
        self._hud_bottom_cache: dict[tuple[int, int], Image.Image] = {}

    @staticmethod
    def _signed_area(points):
        return 0.5 * sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:])
        )

    @staticmethod
    def _scale_matrix(size: tuple[int, int]) -> Matrix:
        width, height = size
        return Matrix(sx=width / STAGE_SIZE[0], sy=height / STAGE_SIZE[1])

    def _draw_shape(
        self,
        image: Image.Image,
        shape: Shape,
        matrix: Matrix,
        alpha_scale: float = 1.0,
    ):
        """Rasterize Flash fills/strokes using the source shape fill rule."""
        for fi in range(1, len(shape.fills)):
            color = shape.fills[fi]
            if color is None or color[3] == 0:
                continue
            contours = []
            all_x, all_y = [], []
            for loop in shape.loops(fi):
                pts = [matrix.apply(x, y) for x, y in loop]
                if len(pts) < 3:
                    continue
                contours.append((pts, 1 if self._signed_area(pts) >= 0 else -1))
                all_x.extend(p[0] for p in pts)
                all_y.extend(p[1] for p in pts)
            if not contours:
                continue
            left = max(0, int(np.floor(min(all_x))) - 1)
            top = max(0, int(np.floor(min(all_y))) - 1)
            right = min(image.width, int(np.ceil(max(all_x))) + 2)
            bottom = min(image.height, int(np.ceil(max(all_y))) + 2)
            if right <= left or bottom <= top:
                continue
            acc = np.zeros((bottom - top, right - left), dtype=np.int16)
            for pts, sign in contours:
                mask = Image.new("L", (right - left, bottom - top), 0)
                md = ImageDraw.Draw(mask)
                md.polygon([(x - left, y - top) for x, y in pts], fill=1)
                acc += np.asarray(mask, dtype=np.int16) * sign
            alpha = int(round(color[3] * max(0.0, min(1.0, alpha_scale))))
            occupied = (acc != 0) if shape.non_zero_winding else ((np.abs(acc) & 1) != 0)
            mask_arr = np.where(occupied, alpha, 0).astype(np.uint8)
            mask = Image.fromarray(mask_arr, mode="L")
            patch = Image.new(
                "RGBA", (right - left, bottom - top), (color[0], color[1], color[2], 255)
            )
            image.paste(patch, (left, top), mask)

        draw = ImageDraw.Draw(image, "RGBA")
        for edge in shape.edges:
            if edge.line <= 0 or edge.line >= len(shape.lines):
                continue
            style = shape.lines[edge.line]
            if style is None or style.color[3] == 0 or style.width <= 0:
                continue
            pts = shape._sample(edge, 24 if edge.kind == "curve" else 1)
            pts = [matrix.apply(x, y) for x, y in pts]
            sx = (matrix.sx * matrix.sx + matrix.r1 * matrix.r1) ** 0.5
            sy = (matrix.r0 * matrix.r0 + matrix.sy * matrix.sy) ** 0.5
            scale = (sx + sy) * 0.5
            width = max(1, int(round(style.width * scale)))
            a = int(round(style.color[3] * max(0.0, min(1.0, alpha_scale))))
            col = (style.color[0], style.color[1], style.color[2], a)
            draw.line(pts, fill=col, width=width, joint="curve")

    def _draw_symbol(
        self,
        image: Image.Image,
        inst: Instance,
        t: int = 0,
        alpha_scale: float = 1.0,
        output_transform: Matrix | None = None,
    ):
        for shape, m in self.data.swf.flatten(inst.symbol, inst.matrix, t):
            if output_transform is not None:
                m = output_transform.then(m)
            self._draw_shape(image, shape, m, alpha_scale)

    def _draw_enemies_fast(
        self,
        image: Image.Image,
        core: WorldsHardestGameCore,
        output_transform: Matrix | None = None,
    ):
        draw = ImageDraw.Draw(image, "RGBA")
        for shape, m in self.data.swf.flatten(
            core.defn.enemies.symbol, core.defn.enemies.matrix, core.enemy_frame
        ):
            if output_transform is not None:
                m = output_transform.then(m)
            if shape.sid not in (80, 265, 353):
                self._draw_shape(image, shape, m, 1.0)
                continue
            cx, cy = m.apply(0.0, 0.0)
            sx = (m.sx * m.sx + m.r1 * m.r1) ** 0.5
            sy = (m.r0 * m.r0 + m.sy * m.sy) ** 0.5
            scale = (sx + sy) * 0.5
            outer = 6.5 * scale
            inner = 3.5 * scale
            outer_col = shape.fills[1] or (0, 0, 0, 255)
            inner_col = shape.fills[2] or outer_col
            draw.ellipse((cx - outer, cy - outer, cx + outer, cy + outer), fill=outer_col)
            draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=inner_col)

    def _draw_coin_fast(
        self,
        image: Image.Image,
        coin: Instance,
        output_transform: Matrix | None = None,
    ):
        draw = ImageDraw.Draw(image, "RGBA")
        leaves = tuple(self.data.swf.flatten(coin.symbol, coin.matrix, 0))
        if len(leaves) == 1 and leaves[0][0].sid == 114:
            shape, m = leaves[0]
            if output_transform is not None:
                m = output_transform.then(m)
            cx, cy = m.apply(0.0, 0.0)
            sx = (m.sx * m.sx + m.r1 * m.r1) ** 0.5
            sy = (m.r0 * m.r0 + m.sy * m.sy) ** 0.5
            scale = (sx + sy) * 0.5
            outer = 6.5 * scale
            inner = 3.5 * scale
            black = shape.fills[2] or (0, 0, 0, 255)
            yellow = shape.fills[1] or (255, 255, 0, 255)
            draw.ellipse((cx - outer, cy - outer, cx + outer, cy + outer), fill=black)
            draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=yellow)
        else:
            self._draw_symbol(image, coin, 0, output_transform=output_transform)

    def clear_static_cache(self) -> None:
        self._static_cache.clear()
        self._hud_cache.clear()
        self._hud_top_cache.clear()
        self._hud_bottom_cache.clear()

    def _static_level(self, level: int, size: tuple[int, int]) -> Image.Image:
        key = (level, int(size[0]), int(size[1]))
        cached = self._static_cache.get(key)
        if cached is not None:
            return cached.copy()

        d = self.data.level(level)
        transform = None if size == STAGE_SIZE else self._scale_matrix(size)
        im = Image.new("RGBA", size, (180, 181, 254, 255))
        self._draw_symbol(im, d.background, 0, output_transform=transform)
        for _, check in sorted(d.checks.items(), key=lambda kv: kv[1].depth):
            self._draw_symbol(im, check, 0, output_transform=transform)
        self._draw_symbol(im, d.walls, 0, output_transform=transform)

        self._static_cache[key] = im.copy()
        return im

    def _hud_overlay(self, level: int, size: tuple[int, int]) -> Image.Image:
        key = (level, int(size[0]), int(size[1]))
        cached = self._hud_cache.get(key)
        if cached is not None:
            return cached
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        sy = size[1] / STAGE_SIZE[1]
        draw = ImageDraw.Draw(overlay, "RGBA")
        top_h = max(1, int(round(25 * sy)))
        bottom_y = int(round(375 * sy))
        draw.rectangle((0, 0, size[0], top_h), fill=(0, 0, 0, 255))
        draw.rectangle((0, bottom_y, size[0], size[1]), fill=(0, 0, 0, 255))
        if size[1] >= 120:
            draw.text(self._scaled_point(252, 6, size), f"{level}/30", fill=(255,255,255,255))
        self._hud_cache[key] = overlay
        return overlay


    def _hud_top_strip(
        self, core: WorldsHardestGameCore, size: tuple[int, int]
    ) -> Image.Image:
        """Opaque final top HUD, cached without changing its rasterization."""
        key = (
            core.level, int(size[0]), int(size[1]),
            int(core.deaths), int(core.current_coins),
        )
        cached = self._hud_top_cache.get(key)
        if cached is not None:
            return cached

        sx = size[0] / STAGE_SIZE[0]
        sy = size[1] / STAGE_SIZE[1]
        top_h = max(1, int(round(25 * sy)))
        # ImageDraw.rectangle includes the bottom endpoint, so the old
        # (0, 0, width, top_h) rectangle occupies top_h + 1 rows.
        strip_h = min(size[1], top_h + 1)
        strip = Image.new("RGBA", (size[0], strip_h), (0, 0, 0, 255))
        draw = ImageDraw.Draw(strip, "RGBA")

        if size[1] >= 120:
            draw.text(
                self._scaled_point(252, 6, size),
                f"{core.level}/30",
                fill=(255, 255, 255, 255),
            )
            draw.text(
                self._scaled_point(420, 6, size),
                f"DEATHS: {core.deaths}",
                fill=(255, 255, 255, 255),
            )
            if COIN_COUNTS[core.level - 1]:
                draw.text(
                    self._scaled_point(8, 6, size),
                    f"COINS: {core.current_coins}/{COIN_COUNTS[core.level - 1]}",
                    fill=(255, 255, 255, 255),
                )
        elif COIN_COUNTS[core.level - 1]:
            x = max(1, int(round(8 * sx)))
            y = max(1, int(round(6 * sy)))
            if 0 <= y < strip_h:
                draw.point((x, y), fill=(255, 255, 255, 255))

        self._hud_top_cache[key] = strip
        return strip

    def _hud_bottom_strip(self, size: tuple[int, int]) -> tuple[int, Image.Image]:
        key = (int(size[0]), int(size[1]))
        cached = self._hud_bottom_cache.get(key)
        sy = size[1] / STAGE_SIZE[1]
        bottom_y = int(round(375 * sy))
        bottom_y = max(0, min(size[1], bottom_y))
        if cached is None:
            cached = Image.new(
                "RGBA", (size[0], size[1] - bottom_y), (0, 0, 0, 255)
            )
            self._hud_bottom_cache[key] = cached
        return bottom_y, cached

    def _apply_cached_hud(
        self, im: Image.Image, core: WorldsHardestGameCore, size: tuple[int, int]
    ) -> None:
        # The source HUD bars are fully opaque. Replacing those strips after
        # drawing the game scene is exactly equivalent to alpha-compositing the
        # old full-frame HUD overlay and then drawing its dynamic text.
        im.paste(self._hud_top_strip(core, size), (0, 0))
        bottom_y, bottom = self._hud_bottom_strip(size)
        if bottom.height:
            im.paste(bottom, (0, bottom_y))

    @staticmethod
    def _scaled_point(x: float, y: float, size: tuple[int, int]) -> tuple[int, int]:
        return (
            int(round(x * size[0] / STAGE_SIZE[0])),
            int(round(y * size[1] / STAGE_SIZE[1])),
        )

    def _draw_dynamic_hud(self, im: Image.Image, core: WorldsHardestGameCore, size: tuple[int, int]) -> None:
        """Draw a lightweight HUD appropriate for the selected framebuffer.

        At 550x400 this matches the previous renderer.  At small RL sizes the
        text is intentionally tiny; the important bars remain pixel-identical
        in layout while avoiding a full-resolution temporary image + resize.
        """
        draw = ImageDraw.Draw(im, "RGBA")
        sx = size[0] / STAGE_SIZE[0]
        sy = size[1] / STAGE_SIZE[1]
        # Pillow's default bitmap font is already small; only draw text if the
        # target has enough vertical resolution for it to be meaningful.
        if size[1] >= 120:
            draw.text(
                self._scaled_point(420, 6, size),
                f"DEATHS: {core.deaths}",
                fill=(255, 255, 255, 255),
            )
            if COIN_COUNTS[core.level - 1]:
                draw.text(
                    self._scaled_point(8, 6, size),
                    f"COINS: {core.current_coins}/{COIN_COUNTS[core.level - 1]}",
                    fill=(255, 255, 255, 255),
                )
        else:
            # Preserve a few bright HUD pixels as a coarse indicator without
            # spending time rasterizing text that would be unreadable after a
            # 5x reduction anyway.
            if COIN_COUNTS[core.level - 1]:
                x = max(1, int(round(8 * sx)))
                y = max(1, int(round(6 * sy)))
                draw.point((x, y), fill=(255, 255, 255, 255))

    def _render_rgba(
        self,
        core: WorldsHardestGameCore,
        size: tuple[int, int],
    ) -> Image.Image:
        """Render the scene into a Pillow RGBA image.

        Keeping the Pillow image alive lets the legacy-exact resize path resize
        it *before* converting to NumPy.  That removes a full-resolution
        Pillow->NumPy->Pillow round trip while preserving the exact rasterization
        and resize operations used by the old training pipeline.
        """
        size = (int(size[0]), int(size[1]))
        if size[0] < 1 or size[1] < 1:
            raise ValueError("render size must be positive")

        transform = None if size == STAGE_SIZE else self._scale_matrix(size)
        im = self._static_level(core.level, size)
        d = core.defn
        self._draw_enemies_fast(im, core, output_transform=transform)
        for coin, taken in zip(d.coins, core.collected):
            if not taken:
                self._draw_coin_fast(im, coin, output_transform=transform)

        player_matrix = Matrix(tx=core.player_x, ty=core.player_y)
        if transform is not None:
            player_matrix = transform.then(player_matrix)
        player_inst = Instance(86, player_matrix, "player", 999)
        # player_matrix is already output-space when scaled, so do not apply
        # the output transform a second time.
        self._draw_symbol(
            im,
            player_inst,
            0,
            max(0.0, min(1.0, core.alpha / 100.0)),
        )
        # HUD is top-most. Cache its final opaque strips, including dynamic
        # text, so font rasterization is paid only when counters change.
        self._apply_cached_hud(im, core, size)
        return im

    def rgb(
        self,
        core: WorldsHardestGameCore,
        size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Render RGB at source resolution or directly at ``size``.

        The public output is intentionally unchanged from the previous renderer.
        """
        if size is None:
            size = STAGE_SIZE
        im = self._render_rgba(core, size)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def rgb_legacy_resized(
        self,
        core: WorldsHardestGameCore,
        size: tuple[int, int] = (110, 80),
        method: str = "box",
    ) -> np.ndarray:
        """Return the old full-resolution->resize observation, byte-for-byte.

        Legacy training did::

            full = renderer.rgb(core)
            small = Image.fromarray(full).resize(size, resample=...)

        This method performs the same source-resolution rasterization, RGB
        conversion and Pillow resize, but keeps the intermediate image inside
        Pillow.  It therefore avoids materializing the 550x400 RGB NumPy array
        and immediately reconstructing a Pillow image from it.
        """
        width, height = int(size[0]), int(size[1])
        if width < 1 or height < 1:
            raise ValueError("observation size must be positive")
        try:
            resample = {
                "nearest": Image.Resampling.NEAREST,
                "bilinear": Image.Resampling.BILINEAR,
                "box": Image.Resampling.BOX,
            }[method]
        except KeyError as exc:
            raise ValueError("method must be nearest, bilinear, or box") from exc

        # IMPORTANT: convert to RGB at 550x400 *before* resizing.  This matches
        # the original full-resolution + resize pixels exactly, including partially transparent death
        # frames; resizing RGBA first would not be guaranteed identical.
        image = self._render_rgba(core, STAGE_SIZE).convert("RGB")
        if (width, height) != STAGE_SIZE:
            image = image.resize((width, height), resample=resample)
        return np.asarray(image, dtype=np.uint8)
