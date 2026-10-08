# collage

Drop in images or PDFs, cut them into sections, scatter and layer the sections into a new picture, export at print resolution.

```
py -m pip install -r requirements.txt
py server.py            # opens http://127.0.0.1:8765  (py server.py 9000 for another port)
py test_collage.py
```

Python 3.10+, two dependencies (Pillow, PyMuPDF). Everything runs locally; the server only listens on 127.0.0.1.

## How it works

1. **Source.** Add any number of images or PDFs (drag onto the window too). The physical size is read from the PDF page or the image's dpi tag. If an image has no dpi tag the size is assumed at 300 dpi and flagged; type the real width and the height follows. **Random from folder:** paste a folder path, say how many, press *pick*: that many random images/PDFs from the folder (not its sub-folders) replace the previous random pick; files you added yourself stay.
2. **Cut.** Width/height sliders (or typed physical sizes) set the section size as a share of the source, which sets the grid (25% = 4 columns). *Portion of each file* below 100% cuts only random square portions of each file (*portions per file* of them), each cut into the same grid; sections stay in the part of the canvas their portion came from.
3. **Collage.** Each section goes back where it was cut, then moves. Every slider value can be typed.

   | slider | does |
   |---|---|
   | scale | section size on the canvas, relative to its real physical size |
   | size variance | random spread around that scale |
   | rotation | max random tilt, degrees |
   | scatter | 0 = home position, 100% = anywhere on the canvas |
   | density | share of sections used per layer |
   | layers | how many times the sections are laid down, each on top of the last |
   | opacity | per-section transparency |
   | edge blur | feathers each section's edge into the paper |
   | grid pull / size / irregularity | sections are drawn to the nearest node of a guide grid; irregularity shakes the nodes off the regular lattice |
   | drawing pull | click *draw on preview* and sketch: sections drift toward your lines |
   | seed / paper | reroll the randomness / canvas colour |

   Double-click a slider to reset it. With scatter, variance and rotation at 0 and one layer, the output is the source, rebuilt exactly.
4. **Tone.** Contrast, black/white point and gamma, then a tone curve (click the curve to add a point, drag to move it, double-click to remove). Applied to the sources, so preview and export match.
5. **Output.** Size (in / cm / mm), dpi, and `png` (lossless), `tif` (lossless, Deflate) or `pdf` (page of the exact physical size holding the full-resolution image, lossless). Output is capped at 400 MP.

   - **`map`** exports only the *migration map*: one vector PDF the size of the canvas with four layers you can switch on and off in a PDF viewer. *1 initial*: every section's outline where it was cut. *2 migration*: lines joining each section's corners to its new corners (so move, turn and scale read at once) and an arrow along each centre's path. *3 final*: where each section landed. Each layer has its own tone (colour) and opacity, plus the paper colour; choosing `map` shows it on the stage while you adjust.
   - **also save portion linework** (when *portion of each file* is below 100%) adds a vector PDF, one page per source at its physical size, outlining the selected portions with their numbers and the cut grid inside. With it ticked the download is a zip of the main output and `collage-portions.pdf`.

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
- The migration map and portion linework are vector, but the collage itself is raster, and PDF sources are rasterised, not kept as vectors. They are rendered at the resolution the sections will actually have (72-600 dpi).
- Tone is one curve applied to all channels together (no separate R/G/B curves). The drawing pull and grid use at most 400 sampled points / 48 grid rows.
- A 24 x 36 in, 300 dpi export (78 MP) with ~200 rotated sections takes about 10-15 s and ~1.3 GB of RAM; rotation dominates.
