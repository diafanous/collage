"""py test_collage.py  (also works under pytest)"""
import io
import json
import pathlib
import struct
import tempfile
import threading
import zipfile

import pymupdf
from PIL import Image, ImageChops

import collage as C
import server

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


def test_source_size_is_json_serialisable_for_every_format():
    for fmt in ("PNG", "TIFF", "JPEG"):   # TIFF dpi arrives as a rational, which json can't encode
        buf = io.BytesIO()
        Image.new("RGB", (300, 150), "red").save(buf, fmt, dpi=(300, 300))
        s = C.open_source("x", buf.getvalue())
        assert json.loads(json.dumps(dict(w=s.w_in, aspect=s.aspect)))["w"] == 1.0, fmt


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
    assert [x[6] for x in moved] == [x[6] for x in one]                    # scatter doesn't reshuffle angles
    assert moved != one


def test_exports_carry_size_and_dpi():
    store = store_of(C.open_source("a", png(400, 300, dpi=200)[1]))
    raw = {**FLAT, "rot": 10, "scatter": 0.3, "w": 2, "h": 1.5, "dpi": 200}
    specs = [{"id": "0", "w": 2}]

    out = Image.open(io.BytesIO(C.export(raw, specs, store, "png")[0]))
    assert out.size == (400, 300) and round(out.info["dpi"][0]) == 200

    out = Image.open(io.BytesIO(C.export(raw, specs, store, "tif")[0]))
    assert out.size == (400, 300) and round(out.info["dpi"][0]) == 200 and out.mode == "RGB"

    doc = pymupdf.open(stream=C.export(raw, specs, store, "pdf")[0], filetype="pdf")
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


def two_sections():
    """A 4in x 3in white source on an 8 x 6 canvas."""
    store = store_of(C.open_source("a", png(400, 300, dpi=100, color="white")[1]))
    return store, [{"id": "0", "w": 4}], {**FLAT, "w": 8, "h": 6, "dpi": 50}


def pieces(raw, size=(4, 3)):
    p = C.clean(raw)
    return C.layout(p, [size], p["w"], p["h"])


def test_corners_turn_the_way_pillow_rotates():
    tile = Image.new("RGBA", (100, 60), (255, 255, 255, 255))
    tile.paste((0, 0, 0, 255), (0, 0, 10, 10))                       # black mark in the top-left corner
    rot = tile.rotate(30, Image.BICUBIC, expand=True)
    dark = [(x, y) for y in range(rot.height) for x in range(rot.width)
            if rot.getpixel((x, y))[3] > 200 and rot.getpixel((x, y))[0] < 60]
    mark = (sum(x for x, _ in dark) / len(dark) - rot.width / 2, sum(y for _, y in dark) / len(dark) - rot.height / 2)
    tl = C.corners(0, 0, 100, 60, 30)[0]
    assert abs(mark[0] - tl[0]) < 10 and abs(mark[1] - tl[1]) < 10, (mark, tl)   # mark sits at corner 0, not another


def test_grid_pull_lands_sections_on_the_nodes():
    raw = {**FLAT, "w": 8, "h": 6, "scatter": 0.7, "gs": 1, "gn": 4, "gi": 0}
    on_node = lambda v, cell: abs((v / cell - 0.5) - round(v / cell - 0.5)) < 1e-9       # v = (i + .5) * cell
    for pc in pieces(raw):
        assert on_node(pc[2], 2) and on_node(pc[3], 2), pc                              # 8in / 4 cols, 6in / 3 rows
    assert any(not on_node(pc[2], 2) for pc in pieces({**raw, "gi": 1}))               # irregular grid moves the nodes


def test_drawn_strokes_pull_sections_toward_them():
    line = [[0.9, y / 40] for y in range(41)]                                          # a vertical stroke at 90% across
    raw = {**FLAT, "w": 8, "h": 6, "scatter": 0.8, "pts": line, "draw": 1}
    assert all(abs(pc[2] - 7.2) < 1e-9 for pc in pieces(raw))
    assert any(abs(pc[2] - 7.2) > 0.5 for pc in pieces({**raw, "draw": 0}))            # pull 0 leaves them alone


def test_portions_cut_only_part_of_each_file():
    raw = {**FLAT, "ps": 0.4, "pn": 2}
    part, whole = pieces(raw), pieces({**FLAT, "ps": 1})
    assert len(part) == 2 * 16 and len(whole) == 16
    rects = C._portions(C.clean(raw), 0)
    for pc in part:
        u0, v0, u1, v1 = pc[1]
        assert any(a - 1e-9 <= u0 and u1 <= c + 1e-9 and b - 1e-9 <= v0 and v1 <= d + 1e-9 for a, b, c, d in rects)
    assert rects == C._portions(C.clean(raw), 0)                                       # stable for a seed


def test_tone_lut():
    assert C.tone_lut(C.clean({})) == C.IDENT
    hi = C.tone_lut(C.clean({"con": 1}))
    assert hi[64] < 64 and hi[192] > 192                                               # contrast spreads
    lv = C.tone_lut(C.clean({"lb": 100, "lw": 200}))
    assert lv[100] == 0 and lv[200] == 255 and abs(lv[150] - 127.5) < 1.5
    assert C.tone_lut(C.clean({"lg": 2}))[64] > 64                                     # gamma > 1 lifts the mids
    inv = list(range(255, -1, -1))
    assert C.tone_lut(C.clean({"curve": inv}))[0] == 255                               # the curve is applied
    assert C.clean({"curve": [1, 2, 3]})["curve"] == C.IDENT                           # wrong-length curve ignored
    store = store_of(C.open_source("w", png(50, 50, 50, "white")[1]))
    _, im = C.compose({**FLAT, "w": 1, "h": 1, "dpi": 50, "curve": inv}, [{"id": "0", "w": 1}], store, final=True)
    assert im.getpixel((10, 10)) == (0, 0, 0)                                          # white source comes out black


def test_edge_blur_fades_section_edges_only():
    store, specs, raw = two_sections()
    raw = {**raw, "cw": 1, "ch": 1, "bg": "#000000"}                                   # one section, the whole source
    sharp = C.compose(raw, specs, store, final=True)[1]
    soft = C.compose({**raw, "blur": 0.6}, specs, store, final=True)[1]
    w, h = sharp.size
    assert sharp.getpixel((w // 2, h // 2)) == (255, 255, 255) == soft.getpixel((w // 2, h // 2))   # centre untouched
    edge = (w // 2 - 98, h // 2)                                                       # 2px inside the section's left edge
    assert sharp.getpixel(edge) == (255, 255, 255) and soft.getpixel(edge)[0] < 60


def test_grid_spreads_sections_one_per_cell():
    raw = {**FLAT, "w": 8, "h": 6, "scatter": 0.7, "gs": 1, "gn": 8, "gi": 0}          # 8 x 6 = 48 cells for 16 sections
    spots = [(round(pc[2], 6), round(pc[3], 6)) for pc in pieces(raw)]
    assert len(set(spots)) == 16                                                       # nobody shares a cell
    assert len({round(pc[2], 6) for pc in pieces({**raw, "gn": 3})}) <= 3             # coarse grid: only 3 columns exist


def test_irregular_grid_has_uneven_cells():
    even = C._grid(C.clean({**FLAT, "w": 8, "h": 6, "gn": 6, "gi": 0}), 8, 6)
    odd = C._grid(C.clean({**FLAT, "w": 8, "h": 6, "gn": 6, "gi": 1}), 8, 6)
    widths = lambda xs: [b - a for a, b in zip(xs, xs[1:])]
    assert max(widths(even[0])) - min(widths(even[0])) < 1e-9 and len(even[0]) == 7
    assert max(widths(odd[0])) / min(widths(odd[0])) > 1.5                             # visibly uneven
    assert abs(odd[0][-1] - 8) < 1e-9 and abs(odd[1][-1] - 6) < 1e-9                  # still fills the canvas


def test_preview_overlay_draws_the_grid_only_when_asked():
    store, specs, raw = two_sections()
    raw = {**raw, "pv": 400, "gs": 1, "gn": 4}
    plain = Image.open(io.BytesIO(C.preview(raw, specs, store))).convert("RGB")
    grid = Image.open(io.BytesIO(C.preview(raw, specs, store, show_grid=True))).convert("RGB")
    def orange(im):
        d = im.tobytes()
        return sum(1 for i in range(0, len(d), 3) if d[i] > 200 and 60 < d[i + 1] < 130 and d[i + 2] < 100)
    assert orange(plain) == 0 and orange(grid) > 200
    out = C.export(raw, specs, store, "png")[0]                                        # exports never carry the overlay
    assert orange(Image.open(io.BytesIO(out)).convert("RGB")) == 0


def map_raw():
    store, specs, raw = two_sections()
    return store, specs, {**raw, "scatter": 0.8, "rot": 40, "m1p": "#ffffff", "m2c": "#ff0000", "m2o": 0.5, "m3a": 0.5, "m3p": "#00ff00"}


def test_map_layers_have_their_own_tone_opacity_and_paper():
    store, specs, raw = map_raw()
    pdfs = C.migration_pdfs(raw, specs, store)
    assert sorted(pdfs) == ["collage-1-initial.pdf", "collage-2-migration.pdf", "collage-3-final.pdf", "collage-map-layers.pdf"]
    docs = {n: pymupdf.open(stream=d, filetype="pdf") for n, d in pdfs.items()}
    for d in docs.values():
        assert len(d) == 1 and d[0].rect == pymupdf.Rect(0, 0, 8 * 72, 6 * 72) and d[0].get_images() == []   # vector, canvas-sized
    one, two, three = (docs[f"collage-{n}.pdf"][0].get_drawings() for n in ("1-initial", "2-migration", "3-final"))
    assert one[0]["fill"] == (1.0, 1.0, 1.0) and one[0]["fill_opacity"] == 1                           # layer 1 paper: white, opaque
    assert all(d["fill"] is None or d["fill"] == (1.0, 0.0, 0.0) for d in two)                         # layer 2 has no paper
    assert two[0]["color"] == (1.0, 0.0, 0.0) and abs(two[0]["stroke_opacity"] - 0.5) < 1e-2            # tone + line opacity honoured
    assert three[0]["fill"] == (0.0, 1.0, 0.0) and abs(three[0]["fill_opacity"] - 0.5) < 1e-2          # layer 3 paper: green, half opaque
    assert len(two[0]["items"]) == 16 * 5                                                              # 4 corner lines + 1 path per section


def test_map_shows_the_grid_and_strokes_that_steered_it():
    store, specs, raw = map_raw()
    plain = pymupdf.open(stream=C.migration_pdf(raw, specs, store), filetype="pdf")[0].get_drawings()
    steered = pymupdf.open(stream=C.migration_pdf({**raw, "gs": 1, "gn": 4, "pts": [[0.1, 0.1], [0.9, 0.9]], "draw": 1}, specs, store), filetype="pdf")[0].get_drawings()
    assert len(steered) == len(plain) + 2                                                             # a grid drawing and a stroke drawing


def test_combined_map_layers_switch_off_on_their_own():
    store, specs, raw = map_raw()
    raw = {**raw, "m2a": 0, "m3a": 0}
    pdf = C.migration_pdf(raw, specs, store)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    assert sorted(o["name"] for o in doc.get_ocgs().values()) == ["1 initial", "2 migration", "3 final"]
    assert [x["layer"] for x in doc[0].get_drawings() if x.get("layer")][:2] == ["1 initial", "1 initial"]   # paper + outlines

    def inked(hide):
        d = pymupdf.open(stream=pdf, filetype="pdf")
        for ui in d.layer_ui_configs():
            if ui["text"] in hide:
                d.set_layer_ui_config(ui["number"], 2)                                  # 2 = off
        px = d[0].get_pixmap(dpi=30, alpha=False).samples
        return sum(1 for i in range(0, len(px), 3) if px[i:i + 3] != b"\xff\xff\xff")

    base = inked(())
    assert all(inked((n,)) < base for n in ("1 initial", "2 migration", "3 final"))
    assert inked(("1 initial", "2 migration", "3 final")) == 0
    im = Image.open(io.BytesIO(C.preview({**raw, "pv": 400}, specs, store, as_map=True)))
    assert im.size[0] == 400                                                            # the stage can show it


def test_portion_linework_and_zip():
    store, specs, raw = two_sections()
    raw = {**raw, "ps": 0.5, "pn": 3}
    pdf = pymupdf.open(stream=C.portions_pdf(raw, specs, store), filetype="pdf")
    assert len(pdf) == 1 and pdf[0].rect == pymupdf.Rect(0, 0, 4 * 72, 3 * 72)
    assert sum(1 for d in pdf[0].get_drawings() if d.get("width") == 1.5) == 3         # one heavy outline per portion
    data, name = C.export(raw, specs, store, "png", lines=True)
    assert name == "collage.zip" and sorted(zipfile.ZipFile(io.BytesIO(data)).namelist()) == ["collage-portions.pdf", "collage.png"]
    data, name = C.export(raw, specs, store, "map")
    assert name == "collage-map.zip" and len(zipfile.ZipFile(io.BytesIO(data)).namelist()) == 4
    data, name = C.export(raw, specs, store, "map", lines=True)
    assert len(zipfile.ZipFile(io.BytesIO(data)).namelist()) == 5                       # + portion linework
    try:
        C.export({**raw, "ps": 1}, specs, store, "png", lines=True)
    except ValueError as e:
        assert "portion" in str(e)
    else:
        raise AssertionError("linework without portions")


def test_random_folder_pick():
    with tempfile.TemporaryDirectory() as d:
        for i in range(5):
            (pathlib.Path(d) / f"{i}.png").write_bytes(png(20, 20, 50)[1])
        (pathlib.Path(d) / "notes.txt").write_text("x")
        (pathlib.Path(d) / "sub").mkdir()
        got, total = server.pick_files(f'"{d}"', 3)                                    # quoted, as Explorer's Copy as path gives it
        assert total == 5 and len(got) == 3 and len({f.name for f in got}) == 3 and all(f.suffix == ".png" for f in got)
        assert len(server.pick_files(d, 99)[0]) == 5
        assert len({frozenset(f.name for f in server.pick_files(d, 2)[0]) for _ in range(30)}) > 1   # genuinely random
        for bad in (d + "-missing", d + "/notes.txt"):
            try:
                server.pick_files(bad, 1)
            except ValueError:
                continue
            raise AssertionError(bad)
    with tempfile.TemporaryDirectory() as empty:
        try:
            server.pick_files(empty, 1)
        except ValueError as e:
            assert "no images" in str(e)
        else:
            raise AssertionError("empty folder")


def test_a_busy_port_is_never_shared():
    a = server.Server(("127.0.0.1", 0), server.Handler)
    port = a.server_address[1]
    threading.Thread(target=a.serve_forever, daemon=True).start()
    try:
        assert server.running(port) == server.STAMP                      # a copy of this code is recognised and reused
        try:
            server.Server(("127.0.0.1", port), server.Handler)           # Windows would silently share the port otherwise
        except OSError:
            pass
        else:
            raise AssertionError("a second server bound the same port")
    finally:
        a.shutdown()
        a.server_close()
    assert server.running(port) is None                                  # nothing answers once it has stopped


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
