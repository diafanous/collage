# collage

Drop in images or PDFs, cut them into sections, scatter and layer the sections into a new picture, export at print resolution.

```
py -m pip install -r requirements.txt
py server.py            # opens http://127.0.0.1:8765  (py server.py 9000 for another port)
py test_collage.py
```

Python 3.10+, two dependencies (Pillow, PyMuPDF). Everything runs locally; the server only listens on 127.0.0.1.

## How it works

1. **Source.** Add any number of images or PDFs (drag onto the window too). The physical size is read from the PDF page or the image's dpi tag. If an image has no dpi tag the size is assumed at 300 dpi and flagged; type the real width and the height follows.
2. **Cut.** Width/height sliders set the section size as a share of the source, which sets the grid (25% = 4 columns).
3. **Collage.** Each section goes back where it was cut, then moves.

   | slider | does |
   |---|---|
   | scale | section size on the canvas, relative to its real physical size |
   | size variance | random spread around that scale |
   | rotation | max random tilt, degrees |
   | scatter | 0 = home position, 100% = anywhere on the canvas |
   | density | share of sections used per layer |
   | layers | how many times the sections are laid down, each on top of the last |
   | opacity | per-section transparency |
   | seed / paper | reroll the randomness / canvas colour |

   Double-click a slider to reset it. With scatter, variance and rotation at 0 and one layer, the output is the source, rebuilt exactly.
4. **Output.** Size (in / cm / mm), dpi, and `png` (lossless), `tif` (lossless, Deflate) or `pdf` (page of the exact physical size holding the full-resolution image, lossless). Output is capped at 400 MP.

The preview is rendered by the same engine as the export, at screen size. Sliders drag at half resolution and sharpen on release. Layout is in inches, so the export matches the preview at any dpi.

## Layout

| file | |
|---|---|
| `collage.py` | engine: load, cut, layout, render, export |
| `server.py` | local HTTP server, serves the UI and the font |
| `index.html` | the UI (no build step, no framework) |
| `test_collage.py` | self-checks |
| `fonts/` | linked folder holding `paper-mono-v1.0.zip`; the server reads the font straight from the zip. Not tracked by git. |

## Limits

- 8-bit RGB only, no colour management: 16-bit inputs are scaled to 8-bit, CMYK is converted naively (ignoring ICC profiles), and exports carry no colour profile.
- PDF pages are rasterised, not kept as vectors. They are rendered at the resolution the sections will actually have (72-600 dpi).
- A 24 x 36 in, 300 dpi export (78 MP) with ~200 rotated sections takes about 10-15 s and ~1.3 GB of RAM; rotation dominates.
