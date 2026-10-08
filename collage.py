"""Cut images/PDFs into sections, scatter + layer them on a canvas, export PNG / TIFF / PDF.

All geometry is in inches, so a 1000px preview and a 600 dpi export of the same params are the same picture.
"""
import io
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pymupdf
from PIL import Image, ImageOps

MAX_PIXELS = 400_000_000   # output cap: RGB at the cap is ~1.2 GB
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
ASSUMED_DPI = 300          # images that carry no dpi tag
PREVIEW_SRC = 1600         # long edge of the cached preview copy of each source
POOL = ThreadPoolExecutor(os.cpu_count() or 4)

# key: (min, max, default). Anything the client sends is clamped to this.
SPEC = dict(
    cw=(0.05, 1, 0.25), ch=(0.05, 1, 0.25),                   # section size, fraction of source
    scale=(0.1, 4, 1), var=(0, 0.9, 0.25), rot=(0, 180, 12),  # section scale, scale spread, max rotation (deg)
    scatter=(0, 1, 0.35), density=(0.1, 1, 1),                # drift from home spot, share of sections used
    layers=(1, 8, 2), opacity=(0.05, 1, 1), seed=(0, 9999, 1),
    w=(0.25, 200, 8.5), h=(0.25, 200, 11), dpi=(10, 1200, 300), pv=(200, 3000, 1000),  # canvas in, dpi, preview px
)


def clean(raw):
    p = {k: min(max(float(raw.get(k, d)), lo), hi) for k, (lo, hi, d) in SPEC.items()}
    bg = str(raw.get("bg", ""))
    p["bg"] = bg if re.fullmatch(r"#[0-9a-fA-F]{6}", bg) else "#f4f2ee"
    return p


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


def layout(p, sizes, W, H):
    """Sections bottom to top as (source, (u0, v0, u1, v1), cx, cy, w, h, angle); inches, resolution-independent.

    Every section draws the same six random numbers whatever the sliders say, so dragging one slider
    moves the picture smoothly instead of reshuffling it.
    """
    rng = random.Random(int(p["seed"]))
    cols, rows = max(1, int(1 / p["cw"] + 0.5)), max(1, int(1 / p["ch"] + 0.5))
    out = []
    for _ in range(int(p["layers"])):
        layer = []
        for si, (sw, sh) in enumerate(sizes):
            for r in range(rows):
                for c in range(cols):
                    keep, sv, an, rx, ry, z = (rng.random() for _ in range(6))
                    if keep > p["density"]:
                        continue
                    s = p["scale"] * (1 + p["var"] * (2 * sv - 1))
                    bx, by = (c + 0.5) / cols * W, (r + 0.5) / rows * H   # home spot: same relative place on the canvas
                    box = (c / cols, r / rows, (c + 1) / cols, (r + 1) / rows)
                    layer.append((z, (si, box, bx + (rx * W - bx) * p["scatter"], by + (ry * H - by) * p["scatter"],
                                      sw / cols * s, sh / rows * s, p["rot"] * (2 * an - 1))))
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
        si, (u0, v0, u1, v1), cx, cy, w, h, ang = pc
        im = items[si][0]
        l, r = round((cx - w / 2) * ppi), round((cx + w / 2) * ppi)
        t, b = round((cy - h / 2) * ppi), round((cy + h / 2) * ppi)
        tw, th, mx, my = max(r - l, 1), max(b - t, 1), (l + r) / 2, (t + b) / 2
        reach = (math.hypot(tw, th) if ang else max(tw, th)) / 2
        if mx + reach < 0 or mx - reach > cw or my + reach < 0 or my - reach > ch:
            return None   # off the canvas
        tile = im.resize((tw, th), resample, box=(u0 * im.width, v0 * im.height, u1 * im.width, v1 * im.height))
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
    items = []
    for s in specs:
        src = store[s["id"]]
        pg = min(max(int(s.get("page", 0)), 0), src.pages - 1)
        w = min(max(float(s["w"]), 0.1), 400)
        # a PDF only needs rasterising as sharp as its sections end up on the canvas
        im = raster(src, pg, dpi=min(max(ppi * p["scale"] * (1 + p["var"]), 72), 600)) if final else thumb(src, pg)
        items.append((im, w, w * im.height / im.width))
    return p, render(p, items, ppi, Image.LANCZOS if final else Image.BILINEAR)


def preview(raw, specs, store):
    buf = io.BytesIO()
    compose(raw, specs, store)[1].save(buf, "JPEG", quality=88)
    return buf.getvalue()


def export(raw, specs, store, fmt):
    p, im = compose(raw, specs, store, final=True)
    dpi, buf = round(p["dpi"]), io.BytesIO()
    if fmt == "png":
        im.save(buf, "PNG", dpi=(dpi, dpi), compress_level=3)
    elif fmt == "tif":
        im.save(buf, "TIFF", dpi=(dpi, dpi), compression="tiff_adobe_deflate")
    elif fmt == "pdf":   # lossless Flate image on a page of the exact physical size
        doc = pymupdf.open()
        page = doc.new_page(width=p["w"] * 72, height=p["h"] * 72)
        page.insert_image(page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, im.width, im.height, im.tobytes(), False))
        buf.write(doc.tobytes(deflate=True))
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return buf.getvalue()
