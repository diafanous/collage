"""Cut images/PDFs into sections, scatter + layer them on a canvas, export PNG / TIFF / PDF.

All geometry is in inches, so a 1000px preview and a 600 dpi export of the same params are the same picture.
"""
import io
import math
import os
import random
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pymupdf
from PIL import Image, ImageChops, ImageFilter, ImageOps

MAX_PIXELS = 400_000_000   # output cap: RGB at the cap is ~1.2 GB
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
ASSUMED_DPI = 300          # images that carry no dpi tag
PREVIEW_SRC = 1600         # long edge of the cached preview copy of each source
POOL = ThreadPoolExecutor(os.cpu_count() or 4)
IDENT = list(range(256))

# key: (min, max, default). Anything the client sends is clamped to this.
SPEC = dict(
    cw=(0.05, 1, 0.25), ch=(0.05, 1, 0.25),                   # section size, fraction of source
    ps=(0.1, 1, 1), pn=(1, 6, 2),                             # portion of each file used (1 = all of it), portions per file
    scale=(0.1, 4, 1), var=(0, 0.9, 0.25), rot=(0, 180, 12),  # section scale, scale spread, max rotation (deg)
    scatter=(0, 1, 0.35), density=(0.1, 1, 1),                # drift from home spot, share of sections used
    gn=(2, 24, 8), gs=(0, 1, 0), gi=(0, 1, 0),                # guide grid: nodes across, pull to nearest node, node jitter
    draw=(0, 1, 0.7), blur=(0, 1, 0),                         # pull toward drawn strokes, edge softness
    con=(-1, 1, 0), lb=(0, 254, 0), lw=(1, 255, 255), lg=(0.2, 5, 1),   # contrast, levels black / white / gamma
    layers=(1, 8, 2), opacity=(0.05, 1, 1), seed=(0, 9999, 1),
    w=(0.25, 200, 8.5), h=(0.25, 200, 11), dpi=(10, 1200, 300), pv=(200, 3000, 1000),  # canvas in, dpi, preview px
    m1o=(0, 1, 0.8), m2o=(0, 1, 0.8), m3o=(0, 1, 1),          # migration map: layer opacities
)
COLORS = dict(bg="#f4f2ee", mp="#f4f2ee", m1c="#8a8a90", m2c="#ff5a36", m3c="#18181b")   # canvas, map paper, map layer tones


def clean(raw):
    p = {k: min(max(float(raw.get(k, d)), lo), hi) for k, (lo, hi, d) in SPEC.items()}
    for k, d in COLORS.items():
        v = str(raw.get(k, ""))
        p[k] = v if re.fullmatch(r"#[0-9a-fA-F]{6}", v) else d
    pts = [(min(max(float(x), 0), 1), min(max(float(y), 0), 1)) for x, y in list(raw.get("pts") or [])[:5000]]
    p["pts"] = pts[::max(1, -(-len(pts) // 400))]   # drawn points, thinned to <= 400 so the nearest-point search stays cheap
    c = raw.get("curve")
    p["curve"] = [min(max(round(float(v)), 0), 255) for v in c] if isinstance(c, list) and len(c) == 256 else IDENT
    return p


def tone_lut(p):
    """Contrast, then levels, then the curve, as one 256-entry table."""
    b, g, c = p["lb"] / 255, p["lg"], 2 ** p["con"]
    w = max(p["lw"] / 255, b + 1 / 255)
    lut = []
    for v in range(256):
        x = min(max((v / 255 - 0.5) * c + 0.5, 0), 1)
        x = min(max((x - b) / (w - b), 0), 1) ** (1 / g)
        lut.append(p["curve"][round(x * 255)])
    return lut


def adjust(im, lut):
    if lut == IDENT:
        return im
    return im.point(lut * 3 + (IDENT if im.mode == "RGBA" else []))   # alpha untouched


@dataclass
class Source:
    name: str
    data: bytes
    pdf: bool
    pages: int
    w_in: float          # width read from the file, or ASSUMED_DPI guess
    aspect: float        # height / width
    assumed: bool        # True when w_in is a guess the user should override
    thumbs: dict = field(default_factory=dict)   # page -> preview-size RGB(A)


def _decode(data):
    im = Image.open(io.BytesIO(data))
    dpi = im.info.get("dpi")
    im = ImageOps.exif_transpose(im)
    im.load()
    if im.mode.startswith("I;16"):
        im = im.point(lambda v: v / 256)   # convert() alone clips 16-bit gray to white
    return im.convert("RGBA" if im.has_transparency_data else "RGB"), dpi


def open_source(name, data):
    try:
        if data[:5] == b"%PDF-":
            doc = pymupdf.open(stream=data, filetype="pdf")
            r = doc[0].rect
            src = Source(name, data, True, len(doc), r.width / 72, r.height / r.width, False)
        else:
            im, dpi = _decode(data)
            ok = dpi and float(dpi[0]) > 1   # dpi (1, 1) means "aspect ratio only"
            d = round(float(dpi[0]), 1) if ok else ASSUMED_DPI   # PNG stores px/metre: 300 dpi reads back as 299.9994
            src = Source(name, data, False, 1, im.width / d, im.height / im.width, not ok)
            src.thumbs[0] = im.copy()
            src.thumbs[0].thumbnail((PREVIEW_SRC, PREVIEW_SRC), Image.BILINEAR)
        return src
    except Exception as e:
        raise ValueError(f"can't read {name}") from e


def raster(src, page=0, dpi=None, long_edge=None):
    """One page at dpi / to long_edge px (PDF), or the full-resolution image."""
    if not src.pdf:
        return _decode(src.data)[0]
    pg = pymupdf.open(stream=src.data, filetype="pdf")[page]
    r = pg.rect
    z = dpi / 72 if dpi else long_edge / max(r.width, r.height)
    z = min(z, math.sqrt(150e6 / (r.width * r.height)))   # cap a page at 150 MP
    pix = pg.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def thumb(src, page):
    if page not in src.thumbs:
        src.thumbs[page] = raster(src, page, long_edge=PREVIEW_SRC)
    return src.thumbs[page]


def _portions(p, si):
    """Rectangles (u0, v0, u1, v1) of source si that get cut: the whole file, or random portions of it."""
    if p["ps"] > 0.999:
        return [(0, 0, 1, 1)]
    g, s, out = random.Random(int(p["seed"]) * 131 + si), p["ps"], []
    for _ in range(int(p["pn"])):
        x, y = g.random() * (1 - s), g.random() * (1 - s)
        out.append((x, y, x + s, y + s))
    return out


def _nodes(p, W, H):
    """Guide grid nodes[i][j] in inches: gn columns, square-ish rows, each node pushed off its regular spot by gi."""
    gx, gy = int(p["gn"]), min(48, max(1, round(p["gn"] * H / W)))
    g, dx, dy = random.Random(int(p["seed"]) * 977 + 13), W / gx, H / gy
    return [[((i + 0.5 + p["gi"] * (g.random() - 0.5)) * dx, (j + 0.5 + p["gi"] * (g.random() - 0.5)) * dy)
             for j in range(gy)] for i in range(gx)]


def _nearest_node(nodes, x, y, W, H):
    gx, gy = len(nodes), len(nodes[0])
    i, j = min(max(int(x / W * gx), 0), gx - 1), min(max(int(y / H * gy), 0), gy - 1)
    near = (n for col in nodes[max(i - 1, 0):i + 2] for n in col[max(j - 1, 0):j + 2])   # jitter is under half a cell
    return min(near, key=lambda n: (n[0] - x) ** 2 + (n[1] - y) ** 2)


def layout(p, sizes, W, H):
    """Sections bottom to top as (source, (u0, v0, u1, v1), cx, cy, w, h, angle, home_x, home_y, home_w, home_h).

    Inches, resolution-independent. Every section draws the same six random numbers whatever the sliders say,
    so dragging one slider moves the picture smoothly instead of reshuffling it.
    """
    rng = random.Random(int(p["seed"]))
    cols, rows = max(1, int(1 / p["cw"] + 0.5)), max(1, int(1 / p["ch"] + 0.5))
    nodes = _nodes(p, W, H) if p["gs"] > 0 else None
    pts = [(x * W, y * H) for x, y in p["pts"]] if p["draw"] > 0 else []
    parts = [_portions(p, si) for si in range(len(sizes))]
    out = []
    for _ in range(int(p["layers"])):
        layer = []
        for si, (sw, sh) in enumerate(sizes):
            for a0, b0, a1, b1 in parts[si]:
                fw, fh = (a1 - a0) / cols, (b1 - b0) / rows
                for r in range(rows):
                    for c in range(cols):
                        keep, sv, an, rx, ry, z = (rng.random() for _ in range(6))
                        if keep > p["density"]:
                            continue
                        s = p["scale"] * (1 + p["var"] * (2 * sv - 1))
                        u0, v0 = a0 + c * fw, b0 + r * fh
                        bx, by = (u0 + fw / 2) * W, (v0 + fh / 2) * H   # home: same relative place on the canvas as in the source
                        x, y = bx + (rx * W - bx) * p["scatter"], by + (ry * H - by) * p["scatter"]
                        if nodes:
                            nx, ny = _nearest_node(nodes, x, y, W, H)
                            x, y = x + (nx - x) * p["gs"], y + (ny - y) * p["gs"]
                        if pts:
                            qx, qy = min(pts, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
                            x, y = x + (qx - x) * p["draw"], y + (qy - y) * p["draw"]
                        layer.append((z, (si, (u0, v0, u0 + fw, v0 + fh), x, y, sw * fw * s, sh * fh * s,
                                          p["rot"] * (2 * an - 1), bx, by, sw * fw, sh * fh)))
        out += [pc for _, pc in sorted(layer, key=lambda t: t[0])]
    return out


def _batches(pieces, ppi, budget=60e6):
    """Cap how many finished tiles sit in memory at once."""
    batch, px = [], 0
    for pc in pieces:
        batch.append(pc)
        px += pc[4] * pc[5] * ppi * ppi * 2
        if px >= budget:
            yield batch
            batch, px = [], 0
    if batch:
        yield batch


def render(p, items, ppi, resample):
    """items: [(PIL image, width_in, height_in)] -> RGB canvas of p.w x p.h inches at ppi."""
    W, H = p["w"], p["h"]
    cw, ch = round(W * ppi), round(H * ppi)
    canvas = Image.new("RGB", (cw, ch), p["bg"])
    spin = Image.BICUBIC if resample == Image.LANCZOS else Image.BILINEAR

    def prep(pc):
        si, (u0, v0, u1, v1), cx, cy, w, h, ang = pc[:7]
        im = items[si][0]
        l, r = round((cx - w / 2) * ppi), round((cx + w / 2) * ppi)
        t, b = round((cy - h / 2) * ppi), round((cy + h / 2) * ppi)
        tw, th, mx, my = max(r - l, 1), max(b - t, 1), (l + r) / 2, (t + b) / 2
        reach = (math.hypot(tw, th) if ang else max(tw, th)) / 2
        if mx + reach < 0 or mx - reach > cw or my + reach < 0 or my - reach > ch:
            return None   # off the canvas
        tile = im.resize((tw, th), resample, box=(u0 * im.width, v0 * im.height, u1 * im.width, v1 * im.height))
        rad = min(int(p["blur"] * 0.25 * min(tw, th)), (min(tw, th) - 1) // 2)
        if rad >= 1:   # feather the edge: fade alpha to 0 over ~2*rad px
            soft = Image.new("L", (tw, th), 0)
            soft.paste(255, (rad, rad, tw - rad, th - rad))
            tile = tile.convert("RGBA")
            tile.putalpha(ImageChops.multiply(tile.getchannel("A"), soft.filter(ImageFilter.GaussianBlur(rad / 2))))
        if ang:
            tile = tile.convert("RGBA").rotate(ang, spin, expand=True)   # Pillow premultiplies RGBA: no dark fringe
        mask = tile if tile.mode == "RGBA" else None                       # an RGBA mask pastes by its alpha
        if p["opacity"] < 1:
            a = tile.getchannel("A") if mask else Image.new("L", tile.size, 255)
            mask = a.point(lambda v: v * p["opacity"])
        return tile, mask, (round(mx - tile.width / 2), round(my - tile.height / 2))

    for batch in _batches(layout(p, [(w, h) for _, w, h in items], W, H), ppi):
        for res in POOL.map(prep, batch):
            if res:
                canvas.paste(res[0], res[2], res[1])
    return canvas


def _spec(s, store):
    src = store[s["id"]]
    return src, min(max(int(s.get("page", 0)), 0), src.pages - 1), min(max(float(s["w"]), 0.1), 400)


def compose(raw, specs, store, final=False):
    """raw: params, specs: [{id, w (inches), page}] -> (clean params, RGB image)."""
    p = clean(raw)
    if not specs:
        raise ValueError("add an image or PDF first")
    if final:
        ppi = p["dpi"]
        if p["w"] * p["h"] * ppi * ppi > MAX_PIXELS:
            raise ValueError(f"{p['w'] * ppi:.0f} x {p['h'] * ppi:.0f} px is over the {MAX_PIXELS // 10**6} MP limit")
    else:
        ppi = p["pv"] / max(p["w"], p["h"])
    lut, items = tone_lut(p), []
    for s in specs:
        src, pg, w = _spec(s, store)
        # a PDF only needs rasterising as sharp as its sections end up on the canvas
        im = raster(src, pg, dpi=min(max(ppi * p["scale"] * (1 + p["var"]), 72), 600)) if final else thumb(src, pg)
        items.append((adjust(im, lut), w, w * im.height / im.width))
    return p, render(p, items, ppi, Image.LANCZOS if final else Image.BILINEAR)


def corners(cx, cy, w, h, ang):
    """Corners of a w x h box turned ang degrees counter-clockwise on screen, as Image.rotate does."""
    s, c = math.sin(math.radians(ang)), math.cos(math.radians(ang))
    return [(cx + x * c + y * s, cy - x * s + y * c) for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _sizes(specs, store):
    out = []
    for s in specs:
        src, pg, w = _spec(s, store)
        t = thumb(src, pg)
        out.append((w, w * t.height / t.width))
    return out


def migration_pdf(raw, specs, store):
    """Vector PDF of the canvas, no pixels: toggleable layers paper / 1 initial / 2 migration / 3 final.

    Initial = every section's outline where it was cut; final = where it landed; migration = lines joining
    corresponding corners (they show move, turn and scale at once) plus an arrow along each centre's path.
    """
    p = clean(raw)
    if not specs:
        raise ValueError("add an image or PDF first")
    W, H = p["w"], p["h"]
    pieces = layout(p, _sizes(specs, store), W, H)
    doc = pymupdf.open()
    page = doc.new_page(width=W * 72, height=H * 72)
    oc = [doc.add_ocg(n) for n in ("paper", "1 initial", "2 migration", "3 final")]
    pt = lambda q: (q[0] * 72, q[1] * 72)

    def ink(layer, draw, closed=True, fill=0):   # fill: fill opacity as a share of the layer's opacity
        col, op = _rgb(p[f"m{layer}c"]), p[f"m{layer}o"]
        sh = page.new_shape()
        draw(sh)
        sh.finish(color=col, fill=col if fill else None, width=0.6, closePath=closed, stroke_opacity=op, fill_opacity=op * fill, oc=oc[layer])
        sh.commit()

    def initial(sh):
        seen = set()
        for pc in pieces:
            if pc[7:] not in seen:   # each layer re-uses the same home spots
                seen.add(pc[7:])
                sh.draw_polyline([pt(q) for q in corners(*pc[7:], 0)])

    def final(sh):
        for pc in pieces:
            sh.draw_polyline([pt(q) for q in corners(*pc[2:6], pc[6])])

    def lines(sh):
        for pc in pieces:
            for a, b in zip(corners(*pc[7:], 0), corners(*pc[2:6], pc[6])):
                sh.draw_line(pt(a), pt(b))
            sh.draw_line(pt(pc[7:9]), pt(pc[2:4]))

    def heads(sh):
        for pc in pieces:
            (bx, by), (cx, cy) = pc[7:9], pc[2:4]
            d = math.hypot(cx - bx, cy - by)
            if d < 0.05:   # barely moved: no arrow
                continue
            ux, uy, n = (cx - bx) / d, (cy - by) / d, min(0.12, d / 3)
            sh.draw_polyline([pt((cx, cy)), pt((cx - (ux + 0.4 * uy) * n, cy - (uy - 0.4 * ux) * n)),
                              pt((cx - (ux - 0.4 * uy) * n, cy - (uy + 0.4 * ux) * n))])

    page.draw_rect(page.rect, color=None, fill=_rgb(p["mp"]), oc=oc[0])
    ink(1, initial, fill=0.15)
    ink(2, lines, closed=False)
    ink(2, heads, fill=1)
    ink(3, final, fill=0.15)
    return doc.tobytes(deflate=True)


def portions_pdf(raw, specs, store):
    """Vector linework, one page per source at its physical size: the file's outline, each selected portion
    in a heavy line with its number, and the cut grid inside it."""
    p = clean(raw)
    if not specs:
        raise ValueError("add an image or PDF first")
    if p["ps"] > 0.999:
        raise ValueError("portion linework needs a portion of file below 100%")
    cols, rows = max(1, int(1 / p["cw"] + 0.5)), max(1, int(1 / p["ch"] + 0.5))
    doc = pymupdf.open()
    for si, (w, h) in enumerate(_sizes(specs, store)):
        page = doc.new_page(width=w * 72, height=h * 72)
        page.draw_rect(page.rect, color=(0.6, 0.6, 0.6), width=0.5)
        for n, (a0, b0, a1, b1) in enumerate(_portions(p, si), 1):
            x0, y0, x1, y1 = a0 * w * 72, b0 * h * 72, a1 * w * 72, b1 * h * 72
            for c in range(1, cols):
                page.draw_line((x0 + (x1 - x0) * c / cols, y0), (x0 + (x1 - x0) * c / cols, y1), color=(0.45, 0.45, 0.45), width=0.3)
            for r in range(1, rows):
                page.draw_line((x0, y0 + (y1 - y0) * r / rows), (x1, y0 + (y1 - y0) * r / rows), color=(0.45, 0.45, 0.45), width=0.3)
            page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=1.5)
            size = max(6, min(w, h) * 72 * 0.04)
            page.insert_text((x0 + size * 0.4, y0 + size * 1.2), str(n), fontsize=size)
    return doc.tobytes(deflate=True)


def preview(raw, specs, store, as_map=False):
    if as_map:   # the map is a PDF: rasterise its page so the stage can show it
        page = pymupdf.open(stream=migration_pdf(raw, specs, store), filetype="pdf")[0]
        z = clean(raw)["pv"] / max(page.rect.width, page.rect.height)
        return page.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False).tobytes("jpeg", jpg_quality=90)
    buf = io.BytesIO()
    compose(raw, specs, store)[1].save(buf, "JPEG", quality=88)
    return buf.getvalue()


def export(raw, specs, store, fmt, lines=False):
    """-> (file bytes, file name). fmt: png / tif / pdf, or map for the migration PDF alone.
    With lines=True the result is a zip of the file plus the portion linework."""
    if fmt not in ("png", "tif", "pdf", "map"):
        raise ValueError(f"unknown format {fmt!r}")
    extra = portions_pdf(raw, specs, store) if lines else None   # cheap, so a bad request fails before the big render
    if fmt == "map":
        data, name = migration_pdf(raw, specs, store), "collage-map.pdf"
    else:
        p, im = compose(raw, specs, store, final=True)
        dpi, buf = round(p["dpi"]), io.BytesIO()
        if fmt == "png":
            im.save(buf, "PNG", dpi=(dpi, dpi), compress_level=3)
        elif fmt == "tif":
            im.save(buf, "TIFF", dpi=(dpi, dpi), compression="tiff_adobe_deflate")
        else:   # lossless Flate image on a page of the exact physical size
            doc = pymupdf.open()
            page = doc.new_page(width=p["w"] * 72, height=p["h"] * 72)
            page.insert_image(page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, im.width, im.height, im.tobytes(), False))
            buf.write(doc.tobytes(deflate=True))
        data, name = buf.getvalue(), f"collage.{fmt}"
    if extra:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as z:   # stored, not deflated: PNG/TIFF/PDF are already compressed
            z.writestr(name, data)
            z.writestr("collage-portions.pdf", extra)
        return zbuf.getvalue(), "collage.zip"
    return data, name
