from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import struct
import zlib
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Matrix:
    sx: float = 1.0
    sy: float = 1.0
    r0: float = 0.0
    r1: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return self.sx * x + self.r0 * y + self.tx, self.r1 * x + self.sy * y + self.ty

    def inverse_apply(self, x: float, y: float) -> tuple[float, float]:
        x -= self.tx
        y -= self.ty
        det = self.sx * self.sy - self.r0 * self.r1
        if abs(det) < 1e-12:
            return 1e30, 1e30
        return ((self.sy * x - self.r0 * y) / det, (-self.r1 * x + self.sx * y) / det)

    def then(self, child: "Matrix") -> "Matrix":
        """Return transform equivalent to self(child(point))."""
        return Matrix(
            sx=self.sx * child.sx + self.r0 * child.r1,
            r0=self.sx * child.r0 + self.r0 * child.sy,
            tx=self.sx * child.tx + self.r0 * child.ty + self.tx,
            r1=self.r1 * child.sx + self.sy * child.r1,
            sy=self.r1 * child.r0 + self.sy * child.sy,
            ty=self.r1 * child.tx + self.sy * child.ty + self.ty,
        )


IDENTITY = Matrix()


class Bits:
    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.bit = pos * 8

    def ub(self, n: int) -> int:
        value = 0
        for _ in range(n):
            byte_i = self.bit >> 3
            bit_i = 7 - (self.bit & 7)
            value = (value << 1) | ((self.data[byte_i] >> bit_i) & 1)
            self.bit += 1
        return value

    def sb(self, n: int) -> int:
        value = self.ub(n)
        if n and value >> (n - 1):
            value -= 1 << n
        return value

    def align(self) -> None:
        self.bit = (self.bit + 7) // 8 * 8

    @property
    def pos(self) -> int:
        return self.bit // 8


def _rect(data: bytes, pos: int) -> tuple[tuple[float, float, float, float], int]:
    b = Bits(data, pos)
    n = b.ub(5)
    xmin, xmax, ymin, ymax = [b.sb(n) / 20.0 for _ in range(4)]
    b.align()
    return (xmin, xmax, ymin, ymax), b.pos


def _matrix(data: bytes, pos: int) -> tuple[Matrix, int]:
    b = Bits(data, pos)
    sx = sy = 1.0
    r0 = r1 = 0.0
    if b.ub(1):
        n = b.ub(5)
        sx = b.sb(n) / 65536.0
        sy = b.sb(n) / 65536.0
    if b.ub(1):
        n = b.ub(5)
        # SWF stores RotateSkew0 first, then RotateSkew1.  The transform is
        #   x' = ScaleX*x + RotateSkew1*y + TranslateX
        #   y' = RotateSkew0*x + ScaleY*y + TranslateY
        # Matrix.r0 is the y->x coefficient and Matrix.r1 is the x->y
        # coefficient, so the two serialized values must be assigned crosswise.
        rotate_skew0 = b.sb(n) / 65536.0
        rotate_skew1 = b.sb(n) / 65536.0
        r0 = rotate_skew1
        r1 = rotate_skew0
    n = b.ub(5)
    tx = b.sb(n) / 20.0
    ty = b.sb(n) / 20.0
    b.align()
    return Matrix(sx, sy, r0, r1, tx, ty), b.pos


def _cstr(data: bytes, pos: int) -> tuple[str, int]:
    end = data.find(b"\0", pos)
    if end < 0:
        return data[pos:].decode("latin1", "replace"), len(data)
    return data[pos:end].decode("latin1", "replace"), end + 1


def _tag(data: bytes, pos: int, end: int) -> tuple[int, int, int, int] | None:
    if pos + 2 > end:
        return None
    header = struct.unpack_from("<H", data, pos)[0]
    pos += 2
    code = header >> 6
    length = header & 0x3F
    if length == 0x3F:
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
    return code, length, pos, pos + length


@dataclass(slots=True)
class Edge:
    kind: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    control: tuple[float, float] | None
    fill0: int
    fill1: int
    line: int


@dataclass(frozen=True, slots=True)
class LineStyle:
    width: float
    color: tuple[int, int, int, int]


@dataclass(slots=True)
class Shape:
    sid: int
    bounds: tuple[float, float, float, float]
    fills: list[tuple[int, int, int, int] | None]
    lines: list[LineStyle | None]
    edges: list[Edge]
    non_zero_winding: bool = False

    @staticmethod
    def _point_segment(px, py, x0, y0, x1, y1, eps=1e-7) -> bool:
        dx, dy = x1 - x0, y1 - y0
        cross = (px - x0) * dy - (py - y0) * dx
        if abs(cross) > eps:
            return False
        return min(x0, x1) - eps <= px <= max(x0, x1) + eps and min(y0, y1) - eps <= py <= max(y0, y1) + eps

    @staticmethod
    def _point_segment_distance(px, py, x0, y0, x1, y1) -> float:
        dx, dy = x1 - x0, y1 - y0
        den = dx*dx + dy*dy
        if den <= 1e-20:
            return ((px-x0)**2 + (py-y0)**2) ** 0.5
        t = ((px-x0)*dx + (py-y0)*dy) / den
        t = max(0.0, min(1.0, t))
        qx, qy = x0 + t*dx, y0 + t*dy
        return ((px-qx)**2 + (py-qy)**2) ** 0.5

    @staticmethod
    def _sample(edge: Edge, n: int = 12) -> list[tuple[float, float]]:
        if edge.kind == "line":
            return [edge.p0, edge.p1]
        x0, y0 = edge.p0
        cx, cy = edge.control or edge.p0
        x1, y1 = edge.p1
        out = []
        for i in range(n + 1):
            t = i / n
            u = 1.0 - t
            out.append((u*u*x0 + 2*u*t*cx + t*t*x1, u*u*y0 + 2*u*t*cy + t*t*y1))
        return out

    def hit(self, x: float, y: float) -> bool:
        """Flash-style point/shape hit test used by MovieClip.hitTest(x,y,true).

        SWF DefineShape/2/3 use the even-odd fill rule. DefineShape4 only
        switches to non-zero winding when UsesFillWindingRule is set. For
        point hit testing Flash treats edges between two filled regions as
        interior edges, so they must not affect the occupied-area winding.
        """
        xmin, xmax, ymin, ymax = self.bounds
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        winding = 0
        for edge in self.edges:
            has0 = edge.fill0 > 0
            has1 = edge.fill1 > 0
            # An edge with fill on both sides is an interior color boundary,
            # not an outer edge of the occupied shape.
            if has0 == has1:
                continue
            pts = self._sample(edge, 32 if edge.kind == "curve" else 1)
            # SWF fillStyle0 has opposite winding from fillStyle1.
            if has0 and not has1:
                pts = list(reversed(pts))
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if self._point_segment(x, y, x0, y0, x1, y1):
                    return True
                if y0 <= y < y1:
                    if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) > 0:
                        winding += 1
                elif y1 <= y < y0:
                    if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) < 0:
                        winding -= 1

        if (winding != 0) if self.non_zero_winding else ((winding & 1) != 0):
            return True

        # Flash shape hit testing includes strokes. Flash also renders a
        # minimum 1px stroke; WHG's gameplay strokes are 3px, so max(...,1)
        # matches the relevant source shapes.
        for edge in self.edges:
            if edge.line <= 0 or edge.line >= len(self.lines):
                continue
            style = self.lines[edge.line]
            if style is None or style.width <= 0:
                continue
            pts = self._sample(edge, 32 if edge.kind == "curve" else 1)
            r = max(style.width, 1.0) * 0.5 + 1e-7
            for (x0,y0),(x1,y1) in zip(pts, pts[1:]):
                if self._point_segment_distance(x,y,x0,y0,x1,y1) <= r:
                    return True
        return False

    def loops(self, fill_i: int) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Approximate fill boundaries as closed loops for rendering."""
        segments: list[list[tuple[float, float]]] = []
        for e in self.edges:
            if e.fill0 != fill_i and e.fill1 != fill_i:
                continue
            pts = self._sample(e, 12 if e.kind == "curve" else 1)
            if e.fill1 == fill_i:
                pts = list(reversed(pts))
            segments.append(pts)
        loops: list[list[tuple[float, float]]] = []
        eps = 1e-5
        def close(a,b): return abs(a[0]-b[0]) < eps and abs(a[1]-b[1]) < eps
        while segments:
            cur = segments.pop(0)
            changed = True
            while changed and not close(cur[0], cur[-1]):
                changed = False
                for i, s in enumerate(segments):
                    if close(cur[-1], s[0]):
                        cur.extend(s[1:]); segments.pop(i); changed=True; break
                    if close(cur[-1], s[-1]):
                        s = list(reversed(s)); cur.extend(s[1:]); segments.pop(i); changed=True; break
            if len(cur) >= 3:
                loops.append(cur)
        return tuple(tuple(p for p in loop) for loop in loops)


@dataclass(slots=True)
class DisplayObject:
    char: int
    matrix: Matrix
    name: str | None = None
    depth: int = 0


@dataclass(slots=True)
class Sprite:
    sid: int
    frame_count: int
    frames: list[dict[int, DisplayObject]]


class SWF:
    """Small SWF8 reader specialized for the vector/timeline features WHG uses."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = self.path.read_bytes()
        if data[:3] == b"CWS":
            data = b"FWS" + data[3:8] + zlib.decompress(data[8:])
        if data[:3] != b"FWS":
            raise ValueError("Only uncompressed/zlib SWF is supported")
        self.data = data
        self.version = data[3]
        self.stage_rect, p = _rect(data, 8)
        self.fps = struct.unpack_from("<H", data, p)[0] / 256.0
        p += 2
        self.frame_count = struct.unpack_from("<H", data, p)[0]
        p += 2
        self.root_start = p
        self.shape_records: dict[int, tuple[int, int, int]] = {}
        self.sprite_records: dict[int, tuple[int, int, int]] = {}
        self.labels: dict[str, int] = {}
        self._index(self.root_start, len(data), is_root=True)

    def _index(self, start: int, end: int, is_root: bool = False) -> None:
        pos, frame = start, 1
        while pos < end:
            t = _tag(self.data, pos, end)
            if not t: break
            code, _, b, e = t
            if e > end: break
            if code == 0: break
            if code == 1:
                frame += 1
            elif code in (2, 22, 32, 83):
                sid = struct.unpack_from("<H", self.data, b)[0]
                self.shape_records[sid] = (code, b, e)
            elif code == 39:
                sid, nf = struct.unpack_from("<HH", self.data, b)
                self.sprite_records[sid] = (nf, b + 4, e)
                self._index(b + 4, e, False)
            elif code == 43 and is_root:
                label, _ = _cstr(self.data, b)
                self.labels[label] = frame
            pos = e

    @staticmethod
    def _skip_cx(data: bytes, q: int) -> int:
        bb = Bits(data, q)
        has_add = bb.ub(1); has_mult = bb.ub(1); n = bb.ub(4)
        comps = 4 if has_add else 3
        if has_mult:
            [bb.sb(n) for _ in range(comps)]
        if has_add:
            [bb.sb(n) for _ in range(comps)]
        bb.align()
        return bb.pos

    def _rgba(self, q: int, alpha: bool) -> tuple[tuple[int,int,int,int], int]:
        if alpha:
            r,g,b,a = self.data[q:q+4]
            return (r,g,b,a), q+4
        r,g,b = self.data[q:q+3]
        return (r,g,b,255), q+3

    def _fill_style(self, q: int, alpha: bool) -> tuple[tuple[int,int,int,int], int]:
        typ = self.data[q]; q += 1
        if typ == 0:
            return self._rgba(q, alpha)
        if typ in (0x10, 0x12, 0x13):
            _, q = _matrix(self.data, q)
            si = self.data[q]; q += 1
            count = si & 0x0F
            first = None
            for _ in range(count):
                q += 1  # ratio
                col, q = self._rgba(q, alpha)
                first = first or col
            if typ == 0x13:
                q += 2
            return first or (255,255,255,255), q
        if typ in (0x40,0x41,0x42,0x43):
            q += 2  # bitmap id
            _, q = _matrix(self.data, q)
            # Gameplay art does not use bitmap-filled strokes.  White is a
            # harmless fallback for unsupported decorative bitmap fills.
            return (255,255,255,255), q
        raise ValueError(f"unsupported fill type {typ:#x}")

    def _fill_array(self, q: int, alpha: bool) -> tuple[list[tuple[int,int,int,int] | None], int]:
        n = self.data[q]; q += 1
        if n == 0xFF:
            n = struct.unpack_from("<H", self.data, q)[0]; q += 2
        out: list[tuple[int,int,int,int] | None] = [None]
        for _ in range(n):
            color, q = self._fill_style(q, alpha)
            out.append(color)
        return out, q

    def _line_array(self, q: int, code: int) -> tuple[list[LineStyle | None], int]:
        n = self.data[q]; q += 1
        if n == 0xFF:
            n = struct.unpack_from("<H", self.data, q)[0]; q += 2
        out: list[LineStyle | None] = [None]
        for _ in range(n):
            width = struct.unpack_from("<H", self.data, q)[0] / 20.0; q += 2
            if code == 83:  # LINESTYLE2
                bb = Bits(self.data, q)
                _start_cap = bb.ub(2)
                join = bb.ub(2)
                has_fill = bb.ub(1)
                bb.ub(1)  # NoHScaleFlag
                bb.ub(1)  # NoVScaleFlag
                bb.ub(1)  # PixelHintingFlag
                bb.ub(5)  # Reserved
                bb.ub(1)  # NoClose
                bb.ub(2)  # EndCapStyle
                bb.align(); q = bb.pos
                if join == 2:
                    q += 2  # MiterLimitFactor FIXED8
                if has_fill:
                    color, q = self._fill_style(q, True)
                else:
                    color, q = self._rgba(q, True)
            else:
                color, q = self._rgba(q, code >= 32)
            out.append(LineStyle(width, color))
        return out, q

    @lru_cache(maxsize=None)
    def shape(self, sid: int) -> Shape:
        code, b, _ = self.shape_records[sid]
        q = b + 2
        bounds, q = _rect(self.data, q)
        shape_flags = 0
        if code == 83:
            _, q = _rect(self.data, q)
            shape_flags = self.data[q]
            q += 1
        alpha = code >= 32
        fills, q = self._fill_array(q, alpha)
        lines, q = self._line_array(q, code)
        bb = Bits(self.data, q)
        nfill = bb.ub(4); nline = bb.ub(4)
        x=y=0; f0=f1=ln=0
        edges: list[Edge] = []
        while True:
            typ = bb.ub(1)
            if typ == 0:
                new = bb.ub(1); linechg = bb.ub(1); f1chg=bb.ub(1); f0chg=bb.ub(1); move=bb.ub(1)
                if not (new or linechg or f1chg or f0chg or move): break
                if move:
                    n = bb.ub(5); x=bb.sb(n); y=bb.sb(n)
                if f0chg: f0=bb.ub(nfill)
                if f1chg: f1=bb.ub(nfill)
                if linechg: ln=bb.ub(nline)
                if new:
                    bb.align(); q=bb.pos
                    nf,q = self._fill_array(q,alpha); nl,q = self._line_array(q,code)
                    fills.extend(nf[1:]); lines.extend(nl[1:])
                    bb=Bits(self.data,q); nfill=bb.ub(4); nline=bb.ub(4)
            else:
                straight=bb.ub(1); nb=bb.ub(4)+2; x0,y0=x,y
                if straight:
                    general=bb.ub(1)
                    if general: dx=bb.sb(nb); dy=bb.sb(nb)
                    else:
                        vert=bb.ub(1)
                        if vert: dx=0; dy=bb.sb(nb)
                        else: dx=bb.sb(nb); dy=0
                    x+=dx; y+=dy
                    edges.append(Edge("line",(x0/20,y0/20),(x/20,y/20),None,f0,f1,ln))
                else:
                    cdx=bb.sb(nb); cdy=bb.sb(nb); adx=bb.sb(nb); ady=bb.sb(nb)
                    cx=x+cdx; cy=y+cdy; x=cx+adx; y=cy+ady
                    edges.append(Edge("curve",(x0/20,y0/20),(x/20,y/20),(cx/20,cy/20),f0,f1,ln))
        # DefineShape4 bit 2 is UsesFillWindingRule / NON_ZERO_WINDING_RULE.
        # Older DefineShape tags always use even-odd.
        return Shape(sid,bounds,fills,lines,edges,bool(shape_flags & 0x04))

    def _parse_place2(self, b: int) -> tuple[int, DisplayObject | None, bool]:
        q=b; flags=self.data[q];q+=1
        depth=struct.unpack_from("<H",self.data,q)[0];q+=2
        has_clip=flags&0x80; has_clip_depth=flags&0x40; has_name=flags&0x20; has_ratio=flags&0x10
        has_cx=flags&8; has_mat=flags&4; has_char=flags&2; move=bool(flags&1)
        char=None; mat=None; name=None
        if has_char: char=struct.unpack_from("<H",self.data,q)[0];q+=2
        if has_mat: mat,q=_matrix(self.data,q)
        if has_cx: q=self._skip_cx(self.data,q)
        if has_ratio: q+=2
        if has_name: name,q=_cstr(self.data,q)
        if has_clip_depth: q+=2
        return depth, (DisplayObject(char,mat or IDENTITY,name,depth) if char is not None else None), move

    @lru_cache(maxsize=None)
    def sprite(self, sid: int) -> Sprite:
        nf,start,end = self.sprite_records[sid]
        display: dict[int,DisplayObject] = {}
        frames: list[dict[int,DisplayObject]]=[]
        pos=start
        while pos<end:
            t=_tag(self.data,pos,end)
            if not t:break
            code,_,b,e=t
            if code==0:break
            if code==1:
                frames.append({d:DisplayObject(o.char,o.matrix,o.name,o.depth) for d,o in display.items()})
            elif code==28:
                depth=struct.unpack_from("<H",self.data,b)[0]
                display.pop(depth,None)
            elif code==26:
                depth,new,move=self._parse_place2(b)
                if new is not None:
                    if move and depth in display:
                        old=display[depth]
                        display[depth]=DisplayObject(new.char,new.matrix if new.matrix!=IDENTITY else old.matrix,new.name or old.name,depth)
                    else:
                        display[depth]=new
                elif move and depth in display:
                    # Need reparse matrix/name from move-only tag.
                    q=b; flags=self.data[q];q+=1; q+=2
                    old=display[depth]
                    mat=old.matrix; name=old.name
                    if flags&2: q+=2
                    if flags&4: mat,q=_matrix(self.data,q)
                    if flags&8:q=self._skip_cx(self.data,q)
                    if flags&0x10:q+=2
                    if flags&0x20:name,q=_cstr(self.data,q)
                    display[depth]=DisplayObject(old.char,mat,name,depth)
            pos=e
        while len(frames)<nf:
            frames.append({d:DisplayObject(o.char,o.matrix,o.name,o.depth) for d,o in display.items()})
        return Sprite(sid,nf,frames[:nf])

    def root_places_at(self, wanted_frame: int) -> list[DisplayObject]:
        pos=self.root_start;frame=1; out=[]
        while pos<len(self.data):
            t=_tag(self.data,pos,len(self.data))
            if not t:break
            code,_,b,e=t
            if code==0:break
            if code==1:
                if frame==wanted_frame: break
                frame+=1
            elif code==26 and frame==wanted_frame:
                _,obj,_=self._parse_place2(b)
                if obj is not None: out.append(obj)
            pos=e
        return out

    def flatten(self, sid: int, matrix: Matrix = IDENTITY, t: int = 0) -> Iterator[tuple[Shape,Matrix]]:
        if sid in self.shape_records:
            yield self.shape(sid),matrix
            return
        spr=self.sprite(sid)
        state=spr.frames[t % spr.frame_count]
        for _,obj in sorted(state.items()):
            yield from self.flatten(obj.char,matrix.then(obj.matrix),t)

    @lru_cache(maxsize=None)
    def symbol_bounds(self,sid:int) -> tuple[float,float,float,float]:
        if sid in self.shape_records:
            return self.shape(sid).bounds
        spr=self.sprite(sid)
        allb=[]
        # Bounds over all parent timeline frames. Nested movie-clips use their
        # authored symbol bounds, which is exactly what Flash hitTest(target)
        # uses for target bounding boxes.
        for state in spr.frames:
            for obj in state.values():
                cb=self.symbol_bounds(obj.char)
                pts=[obj.matrix.apply(x,y) for x in (cb[0],cb[1]) for y in (cb[2],cb[3])]
                allb.append((min(x for x,_ in pts),max(x for x,_ in pts),min(y for _,y in pts),max(y for _,y in pts)))
        if not allb:return (0,0,0,0)
        return min(b[0] for b in allb),max(b[1] for b in allb),min(b[2] for b in allb),max(b[3] for b in allb)
