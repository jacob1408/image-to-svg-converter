# Image → SVG Converter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jacob1408/image-to-svg-converter/blob/main/Image_to_SVG_Converter.ipynb)

Convert up to **100 images per run** into clean **SVG vectors**, with an optional **transparent background** step, right in Google Colab. No install, no account setup beyond a Google login.

Built for stickers, line art, laser-cut designs and flat illustrations (e.g. AI-generated images from Google Flow), ready for Adobe Illustrator, Cricut, Silhouette or laser software.

## How it works

```
Images (upload / Google Drive folder / Drive zip, max 100)
   │
   ├─ Transparent OFF ─────────────────────────────────► Vectorize ─► SVG
   │
   └─ Transparent ON ─► Remove background ─► PNG ─────► Vectorize ─► SVG
```

- **Background removal (`auto`)**
  - Plain/flat backgrounds: exact colour-based removal from the edges. The design is untouched and interior whites are kept.
  - Busy/photo backgrounds: AI removal ([rembg](https://github.com/danielgatis/rembg)), GPU-accelerated when available.
  - Already-transparent PNGs are left as they are.
- **Vectorizing (`auto`)** with [vtracer](https://github.com/visioncortex/vtracer):
  - Black/white line art → clean black-only paths.
  - Colour art → high-fidelity colour trace.
  - Textured/noisy images → simplified automatically, so files stay small.
- **Fast & reliable**
  - Parallel processing on all cores.
  - Per-image timeout with an automatic simplified retry; one bad image never stops the batch.
- **Output**
  - `svg/` plus `transparent_png/`, zipped.
  - Auto-downloaded and optionally saved to your Google Drive.

## Usage

1. Click **Open in Colab** above.
2. *(Optional, for AI background removal)* `Runtime → Change runtime type → T4 GPU`.
3. Set your options in **Step 1 — Settings**.
4. `Runtime → Run all`. Colab will warn that the notebook isn't authored by Google; click **Run anyway**.
5. If the source is `upload`, a file picker appears in Step 4. Select images and/or `.zip` files.

Everything runs in **your own** Colab session. Files only go to your own Google Drive.

## Settings

| Setting | Options | Notes |
|---|---|---|
| `SOURCE` | `upload`, `google_drive_folder`, `google_drive_zip` | Drive paths look like `MyDrive/flow_exports/batch1` |
| `MAKE_TRANSPARENT` | on / off | Removes the background before vectorizing |
| `BG_METHOD` | `auto`, `flat_color`, `ai` | `auto` is recommended |
| `AI_MODEL` | `isnet-general-use`, `birefnet-general`, `u2net` | `birefnet-general` gives the best edges but is slower |
| `BG_TOLERANCE` | 5–60 | Raise it if a light halo remains around edges |
| `VECTOR_MODE` | `auto`, `lineart`, `color`, `poster` | `poster` = fewer colours, cleaner cut shapes |
| `UPSCALE` | `auto`, `1`–`3` | `auto` upscales small images for smoother curves |
| `SAVE_TO_DRIVE` | on / off | Saves to `DRIVE_OUTPUT_PATH` with a timestamp |

## Tips for the cleanest SVGs

- Generate images as **flat vector style, solid colours, no gradients, plain white background**.
- Force `lineart` for laser-cut / single-colour designs and `color` for colour stickers if auto-detection ever guesses wrong.
- Raster-to-vector tracing is always an approximation. Flat art converts near-perfectly, while gradients and soft shading are reduced to colour bands.

## Run locally

```bash
pip install -r requirements.txt
python -c "from vector_engine import run_pipeline; run_pipeline('my_images', 'output', make_transparent=True)"
```

## License

MIT
