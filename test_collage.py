"""py test_collage.py  (also works under pytest)"""
import io
import struct

import pymupdf
from PIL import Image, ImageChops

import collage as C

# neutral settings: sections go back exactly where they were cut
FLAT = dict(cw=0.25, ch=0.25, scale=1, var=0, rot=0, scatter=0, density=1, layers=1, opacity=1, bg="#000000")


def png(w, h, dpi=None, color=None):
    im = Image.new("RGB", (w, h), color) if color else Image.effect_noise((w, h), 80).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG", **({"dpi": (dpi, dpi)} if dpi else {}))
    return im, buf.getvalue()


def pdf(w_pt=612, h_pt=792):
    doc = pymupdf.open()
    page = doc.new_page(width=w_pt, height=h_pt)
    page.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(1, 0, 0))
    return doc.tobytes()


def store_of(*sources):
    return {str(i): s for i, s in enumerate(sources)}


def test_size_read_from_file_or_assumed():
    a = C.open_source("a", png(300, 150, dpi=100)[1])
    assert (a.w_in, a.assumed) == (3.0, False)
    b = C.open_source("b", png(300, 150)[1])
    assert (b.w_in, b.assumed) == (1.0, True)               # 300 px at the assumed 300 dpi
    c = C.open_source("c", pdf())
    assert (c.w_in, c.aspect, c.pages, c.assumed) == (8.5, 792 / 612, 1, False)


def test_16bit_gray_is_scaled_not_clipped():
    ramp = b"".join(struct.pack("<H", x * 257) for _ in range(4) for x in range(256))   # 0 .. 65535
    buf = io.BytesIO()
    Image.frombytes("I;16", (256, 4), ramp).save(buf, "PNG")
    out = C.raster(C.open_source("g", buf.getvalue()))
    assert [out.getpixel((x, 0))[0] for x in (0, 64, 128, 255)] == [0, 64, 128, 255]


def test_unreadable_file_is_rejected():
    for junk in (b"not an image", b"%PDF-1.4 garbage"):
        try:
            C.open_source("x", junk)
        except ValueError:
            continue
        raise AssertionError("accepted junk")


def test_neutral_settings_rebuild_the_source_exactly():
    im, data = png(200, 160, dpi=100)
    store = store_of(C.open_source("a", data))
    _, out = C.compose({**FLAT, "w": 2, "h": 1.6, "dpi": 100}, [{"id": "0", "w": 2}], store, final=True)
    assert ImageChops.difference(im, out).getbbox() is None


def test_rotated_sections_have_no_dark_fringe():
    _, data = png(100, 100, dpi=100, color="white")
    store = store_of(C.open_source("a", data))
    raw = {**FLAT, "bg": "#ffffff", "rot": 40, "scatter": 0.2, "var": 0.3, "w": 1, "h": 1, "dpi": 100}
    for final in (False, True):
        _, out = C.compose({**raw, "pv": 400}, [{"id": "0", "w": 1}], store, final=final)
        assert out.getextrema() == ((255, 255),) * 3, final


def test_layout_is_deterministic_and_stable():
    p = C.clean({**FLAT, "rot": 30, "scatter": 0.5, "var": 0.4, "seed": 7})
    sizes = [(4, 3)]
    one = C.layout(p, sizes, 8, 6)
    assert one == C.layout(p, sizes, 8, 6)
    assert C.layout({**p, "layers": 3}, sizes, 8, 6)[:len(one)] == one      # more layers only add on top
    moved = C.layout({**p, "scatter": 0.9}, sizes, 8, 6)
    assert [x[-1] for x in moved] == [x[-1] for x in one]                    # scatter doesn't reshuffle angles
    assert moved != one


def test_exports_carry_size_and_dpi():
    store = store_of(C.open_source("a", png(400, 300, dpi=200)[1]))
    raw = {**FLAT, "rot": 10, "scatter": 0.3, "w": 2, "h": 1.5, "dpi": 200}
    specs = [{"id": "0", "w": 2}]

    out = Image.open(io.BytesIO(C.export(raw, specs, store, "png")))
    assert out.size == (400, 300) and round(out.info["dpi"][0]) == 200

    out = Image.open(io.BytesIO(C.export(raw, specs, store, "tif")))
    assert out.size == (400, 300) and round(out.info["dpi"][0]) == 200 and out.mode == "RGB"

    doc = pymupdf.open(stream=C.export(raw, specs, store, "pdf"), filetype="pdf")
    assert len(doc) == 1 and doc[0].rect == pymupdf.Rect(0, 0, 144, 108)    # 2 x 1.5 in
    img = doc[0].get_images(full=True)[0]
    assert doc.extract_image(img[0])["width"] == 400                        # full resolution, not downsampled

    try:
        C.export(raw, specs, store, "bmp")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted bmp")


def test_pdf_source_and_multiple_sources():
    store = store_of(C.open_source("p", pdf()), C.open_source("i", png(200, 100, dpi=100)[1]))
    specs = [{"id": "0", "w": 8.5, "page": 0}, {"id": "1", "w": 2}]
    raw = {**FLAT, "layers": 2, "rot": 20, "scatter": 0.5, "w": 4, "h": 5, "dpi": 100}
    _, out = C.compose(raw, specs, store, final=True)
    assert out.size == (400, 500) and out.getextrema() != ((0, 0),) * 3
    assert C.preview({**raw, "pv": 300}, specs, store)[:2] == b"\xff\xd8"    # JPEG


def test_output_size_limit_and_clamping():
    store = store_of(C.open_source("a", png(50, 50, dpi=50)[1]))
    try:
        C.compose({"w": 200, "h": 200, "dpi": 1200}, [{"id": "0", "w": 1}], store, final=True)
    except ValueError as e:
        assert "limit" in str(e)
    else:
        raise AssertionError("no limit")
    p = C.clean({"cw": -5, "layers": 99, "bg": "red; drop table"})
    assert (p["cw"], p["layers"], p["bg"]) == (0.05, 8, "#f4f2ee")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
