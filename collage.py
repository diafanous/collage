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
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

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
    m1o=(0, 1, 0.8), m2o=(0, 1, 0.8), m3o=(0, 1, 1),          # migration map: line opacity per layer
    m1a=(0, 1, 1), m2a=(0, 1, 0), m3a=(0, 1, 0),              # migration map: paper opacity per layer (0 = no paper)
)
COLORS = dict(bg="#f4f2ee", m1c="#8a8a90", m2c="#ff5a36", m3c="#18181b",   # canvas, map tone per layer
              m1p="#f4f2ee", m2p="#f4f2ee", m3p="#f4f2ee")                 # map paper per layer


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


def _grid(p, W, H):
    """Guide grid as cell edges (xs, ys) in inches: gn columns, square-ish rows. gi = 0 gives equal cells;
    more makes columns and rows progressively uneven (up to about twice as wide as their neighbours)."""
    gx, gy = int(p["gn"]), min(48, max(1, round(p["gn"] * H / W)))
    g = random.Random(int(p["seed"]) * 977 + 13)

    def edges(n, total):
        w = [1 + p["gi"] * 0.9 * (2 * g.random() - 1) for _ in range(n)]
        out = [0.0]
        for v in w:
            out.append(out[-1] + v / sum(w) * total)
        return out

    return edges(gx, W), edges(gy, H)


def _cells(xs, ys):
    return [((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2) for i in range(len(xs) - 1) for j in range(len(ys) - 1)]


def layout(p, sizes, W, H):
    """Sections bottom to top as (source, (u0, v0, u1, v1), cx, cy, w, h, angle, home_x, home_y, home_w, home_h).

    Inches, resolution-independent. Every section draws the same six random numbers whatever the sliders say,
    so dragging one slider moves the picture smoothly instead of reshuffling it.
    """
    rng = random.Random(int(p["seed"]))
    cols, rows = max(1, int(1 / p["cw"] + 0.5)), max(1, int(1 / p["ch"] + 0.5))
    cells = _cells(*_grid(p, W, H)) if p["gs"] > 0 else None
    pts = [(x * W, y * H) for x, y in p["pts"]] if p["draw"] > 0 else []
    parts = [_portions(p, si) for si in range(len(sizes))]
    out = []
    for _ in range(int(p["layers"])):
        layer, free = [], list(cells or [])   # each layer hands out every grid cell once, so sections spread over the grid instead of piling on a few
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
                        if cells:
                            free = free or list(cells)   # more sections than cells: start another round
                            nx, ny = min(free, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)
                            free.remove((nx, ny))
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


LAYERS = {1: "1 initial", 2: "2 migration", 3: "3 final"}


def _map_doc(raw, specs, store, layers):
    """Vector PDF of the canvas holding the given map layers (1 initial, 2 migration, 3 final), no pixels.

    Initial = every section's outline where it was cut; final = where it landed; migration = lines joining
    corresponding corners (they show move, turn and scale at once) plus an arrow along each centre's path,
    with the guide grid and drawn strokes in faint lines when they were steering. Each layer has its own tone,
    line opacity and paper (colour + opacity; opacity 0 = no paper). With several layers they are also real PDF
    layers, so a viewer that supports them can switch each on and off.
    """
    p = clean(raw)
    if not specs:
        raise ValueError("add an image or PDF first")
    W, H = p["w"], p["h"]
    pieces = layout(p, _sizes(specs, store), W, H)
    doc = pymupdf.open()
    page = doc.new_page(width=W * 72, height=H * 72)
    oc = {n: doc.add_ocg(LAYERS[n]) if len(layers) > 1 else 0 for n in layers}
    pt = lambda q: (q[0] * 72, q[1] * 72)

    def ink(n, draw, closed=True, fill=0, soft=1):   # fill: fill opacity as a share of the layer's; soft: scales the line opacity
        col, op = _rgb(p[f"m{n}c"]), p[f"m{n}o"]
        sh = page.new_shape()
        draw(sh)
        sh.finish(color=col, fill=col if fill else None, width=0.6, closePath=closed, stroke_opacity=op * soft, fill_opacity=op * fill, oc=oc[n])
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

    def grid(sh):
        xs, ys = _grid(p, W, H)
        for x in xs:
            sh.draw_line(pt((x, 0)), pt((x, H)))
        for y in ys:
            sh.draw_line(pt((0, y)), pt((W, y)))

    def strokes(sh):
        sh.draw_polyline([pt((x * W, y * H)) for x, y in p["pts"]])

    for n in layers:
        if p[f"m{n}a"] > 0:
            page.draw_rect(page.rect, color=None, fill=_rgb(p[f"m{n}p"]), fill_opacity=p[f"m{n}a"], oc=oc[n])
        if n == 1:
            ink(1, initial, fill=0.15)
        elif n == 2:
            if p["gs"] > 0:
                ink(2, grid, closed=False, soft=0.35)
            if p["draw"] > 0 and len(p["pts"]) > 1:
                ink(2, strokes, closed=False, soft=0.6)
            ink(2, lines, closed=False)
            ink(2, heads, fill=1)
        else:
            ink(3, final, fill=0.15)
    return doc.tobytes(deflate=True)


def migration_pdf(raw, specs, store):
    """All three layers in one PDF (also what the stage previews)."""
    return _map_doc(raw, specs, store, (1, 2, 3))


def migration_pdfs(raw, specs, store):
    """{file name: bytes}: each layer as its own PDF, to print or plot and stack, plus the one layered PDF."""
    out = {f"collage-{LAYERS[n].replace(' ', '-')}.pdf": _map_doc(raw, specs, store, (n,)) for n in (1, 2, 3)}
    out["collage-map-layers.pdf"] = migration_pdf(raw, specs, store)
    return out


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


def preview(raw, specs, store, as_map=False, show_grid=False):
    if as_map:   # the map is a PDF: rasterise its page so the stage can show it
        page = pymupdf.open(stream=migration_pdf(raw, specs, store), filetype="pdf")[0]
        z = clean(raw)["pv"] / max(page.rect.width, page.rect.height)
        return page.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False).tobytes("jpeg", jpg_quality=90)
    p, im = compose(raw, specs, store)
    if show_grid:   # guide overlay, preview only: cell edges and a dot where each section will settle
        xs, ys = _grid(p, p["w"], p["h"])
        k, ink = im.width / p["w"], ImageDraw.Draw(im)
        t = max(1, im.width // 400)   # line weight follows the preview size so it stays visible when the stage scales it
        for x in xs:
            ink.line([(x * k, 0), (x * k, im.height)], fill=(255, 90, 54), width=t)
        for y in ys:
            ink.line([(0, y * k), (im.width, y * k)], fill=(255, 90, 54), width=t)
        for x, y in _cells(xs, ys):
            ink.ellipse([x * k - 2 * t, y * k - 2 * t, x * k + 2 * t, y * k + 2 * t], fill=(255, 90, 54))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def export(raw, specs, store, fmt, lines=False):
    """-> (file bytes, file name). fmt: png / tif / pdf, or map for the migration map (a zip of its PDFs).
    With lines=True the portion linework is added, which also makes the result a zip."""
    if fmt not in ("png", "tif", "pdf", "map"):
        raise ValueError(f"unknown format {fmt!r}")
    extra = portions_pdf(raw, specs, store) if lines else None   # cheap, so a bad request fails before the big render
    if fmt == "map":
        files = migration_pdfs(raw, specs, store)
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
        files = {f"collage.{fmt}": buf.getvalue()}
    if extra:
        files["collage-portions.pdf"] = extra
    if len(files) == 1:
        (name, data), = files.items()
        return data, name
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:   # stored, not deflated: PNG/TIFF/PDF are already compressed
        for name, data in files.items():
            z.writestr(name, data)
    return zbuf.getvalue(), "collage-map.zip" if fmt == "map" else "collage.zip"
