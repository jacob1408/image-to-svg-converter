"""Engine: background removal (optional) + raster -> SVG vectorization."""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import vtracer
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_IMAGES = 100

PRESETS = {
    "lineart": dict(colormode="binary", mode="spline", filter_speckle=4,
                    corner_threshold=60, length_threshold=4.0,
                    splice_threshold=45, path_precision=3),
    "color": dict(colormode="color", hierarchical="stacked", mode="spline",
                  filter_speckle=2, color_precision=8, layer_difference=6,
                  corner_threshold=60, length_threshold=4.0,
                  splice_threshold=45, path_precision=3),
    "poster": dict(colormode="color", hierarchical="stacked", mode="spline",
                   filter_speckle=6, color_precision=6, layer_difference=24,
                   corner_threshold=60, length_threshold=5.0,
                   splice_threshold=45, path_precision=2),
}


# ---------------------------------------------------------------- helpers
def load_rgba(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGBA")


def has_transparency(arr):
    return (arr[:, :, 3] < 250).mean() > 0.01


def border_stats(arr, tol):
    """Median border colour and how uniform the border is (0..1)."""
    rgb = arr[:, :, :3].astype(np.int16)
    b = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    med = np.median(b, axis=0)
    uniform = (np.abs(b - med).max(axis=1) <= tol).mean()
    return med, uniform


def flood_remove(arr, tol=24):
    """Remove the flat background connected to the image border.
    Interior areas of the same colour (e.g. white inside a sticker) are kept."""
    arr = arr.copy()
    color, _ = border_stats(arr, tol)
    diff = np.abs(arr[:, :, :3].astype(np.int16) - color.astype(np.int16)).max(axis=2)
    lab, _ = ndimage.label(diff <= tol)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, edge[edge != 0])
    # de-fringe: eat the 1px anti-aliased halo that is still close to bg colour
    bg |= ndimage.binary_dilation(bg) & (diff <= tol * 3)
    arr[bg, 3] = 0
    return arr


def is_lineart(arr):
    """Black/white (grey) artwork with no real colour."""
    small = np.asarray(Image.fromarray(arr, "RGBA").resize((256, 256), Image.NEAREST))
    opaque = small[:, :, 3] > 128
    if opaque.mean() < 0.01:
        return False
    rgb = small[:, :, :3][opaque].astype(np.int16)
    chroma = rgb.max(axis=1) - rgb.min(axis=1)
    gray = rgb.mean(axis=1)
    bw = (chroma < 25) & ((gray < 70) | (gray > 185))
    return bw.mean() >= 0.97


def safe_name(rel):
    return "__".join(Path(rel).with_suffix("").parts)


# ---------------------------------------------------------------- stage 1
def stage_transparent(src, dst, method, tol):
    """Returns 'kept' | 'flat' | 'needs_ai'."""
    arr = np.asarray(load_rgba(src))
    if has_transparency(arr):
        Image.fromarray(arr, "RGBA").save(dst, optimize=False)
        return "kept"
    if method == "ai":
        return "needs_ai"
    _, uniform = border_stats(arr, tol)
    if method == "auto" and uniform < 0.90:
        return "needs_ai"          # busy/photographic background
    Image.fromarray(flood_remove(arr, tol), "RGBA").save(dst)
    return "flat"


def run_ai(jobs, model):
    """jobs: list of (src, dst). Runs on GPU if available, CPU otherwise."""
    try:
        import rembg  # noqa
    except ImportError:
        print("Installing rembg (one time, ~1 min)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rembg[gpu]"], check=True)
    from rembg import new_session, remove
    session = new_session(model)
    errors = {}
    for i, (src, dst) in enumerate(jobs, 1):
        try:
            out = remove(load_rgba(src), session=session, post_process_mask=True)
            out.save(dst)
            print(f"  AI bg {i}/{len(jobs)}: {Path(src).name}")
        except Exception as e:
            errors[src] = str(e)
    return errors


# ---------------------------------------------------------------- stage 2
def noise_level(img):
    """Mean per-pixel texture/noise. Flat art ~0-0.5, heavy texture > 1.5."""
    t = img.convert("RGB")
    t.thumbnail((512, 512))
    a = np.asarray(t).astype(np.int16)
    m = np.asarray(t.filter(ImageFilter.MedianFilter(3))).astype(np.int16)
    return float(np.abs(a - m).mean())


def vectorize(src, dst, mode, upscale, transparent, max_side=2048, safe=False):
    img = load_rgba(src)
    w, h = img.size
    textured = noise_level(img) > 1.5
    if safe or textured:
        # textured/noisy art explodes into millions of tiny paths: denoise + cap size
        rgb = img.convert("RGB").filter(ImageFilter.MedianFilter(3))
        rgb.putalpha(img.getchannel("A"))
        img = rgb
        max_side = 1024 if safe else min(max_side, 1536)
    if upscale == "auto":
        up = 2 if min(w, h) < 1024 else 1
    else:
        up = int(upscale)
    up = max(1, min(up, max_side // max(w, h))) if max(w, h) <= max_side else 1
    if max(w, h) > max_side:            # downscale huge inputs, keep display size
        s = max_side / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    elif up > 1:
        img = img.resize((w * up, h * up), Image.LANCZOS)
    arr = np.asarray(img).copy()

    if transparent:
        # hard alpha edge so the tracer produces clean cut lines
        arr[:, :, 3] = np.where(arr[:, :, 3] >= 128, 255, 0).astype(np.uint8)
    else:
        arr[:, :, 3] = 255

    chosen = mode if mode != "auto" else ("lineart" if is_lineart(arr) else "color")
    if chosen == "color" and (safe or textured):
        chosen = "poster"
    bg_fill = None
    if chosen == "lineart":
        # binary trace = dark shapes only; transparent areas become 'paper'
        rgb = arr[:, :, :3].mean(axis=2)
        rgb[arr[:, :, 3] == 0] = 255
        ink = rgb < 128
        arr = np.zeros_like(arr)
        arr[..., 3] = 255
        arr[~ink, :3] = 255
        if not transparent:
            bg_fill = "#FFFFFF"

    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(arr, "RGBA").save(tmp, compress_level=1)
        vtracer.convert_image_to_svg_py(tmp, dst, **PRESETS[chosen])
    finally:
        os.unlink(tmp)

    svg = Path(dst).read_text(encoding="utf-8")
    W, H = arr.shape[1], arr.shape[0]
    head = f'width="{W}" height="{H}"'
    new_head = f'width="{w}" height="{h}" viewBox="0 0 {W} {H}"'
    svg = svg.replace(head, new_head, 1)
    if bg_fill:
        i = svg.index(">", svg.index("<svg")) + 1
        svg = svg[:i] + f'\n<rect width="{W}" height="{H}" fill="{bg_fill}"/>' + svg[i:]
    Path(dst).write_text(svg, encoding="utf-8")
    return chosen


# ---------------------------------------------------------------- pipeline
def collect_images(input_dir):
    input_dir = Path(input_dir)
    for z in list(input_dir.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent / (z.stem + "_unzipped"))
    files = sorted(p for p in input_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXT
                   and not p.name.startswith((".", "._")) and "__MACOSX" not in p.parts)
    if len(files) > MAX_IMAGES:
        print(f"⚠ {len(files)} images found — processing the first {MAX_IMAGES} only.")
        files = files[:MAX_IMAGES]
    return input_dir, files


def run_pipeline(input_dir, output_dir, make_transparent=False, bg_method="auto",
                 ai_model="isnet-general-use", vector_mode="auto", upscale="auto",
                 tol=24, workers=None, max_side=2048, timeout=90):
    t0 = time.time()
    workers = workers or max(2, os.cpu_count() or 2)
    root, files = collect_images(input_dir)
    if not files:
        raise SystemExit("No images found.")
    out = Path(output_dir)
    svg_dir, png_dir = out / "svg", out / "transparent_png"
    svg_dir.mkdir(parents=True, exist_ok=True)
    if make_transparent:
        png_dir.mkdir(parents=True, exist_ok=True)

    names = {f: safe_name(f.relative_to(root)) for f in files}
    report, errors = {}, {}
    print(f"Found {len(files)} images | transparent={make_transparent} | "
          f"bg={bg_method} | vector={vector_mode} | workers={workers}")

    # ---- Stage 1: transparency
    vec_src = {f: f for f in files}
    if make_transparent:
        ai_jobs = []
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(stage_transparent, str(f), str(png_dir / f"{names[f]}.png"),
                              bg_method, tol): f for f in files}
            for fut in as_completed(futs):
                f = futs[fut]
                try:
                    r = fut.result()
                    if r == "needs_ai":
                        ai_jobs.append(f)
                    else:
                        report[f] = [r]
                except Exception as e:
                    errors[f] = f"transparency: {e}"
        if ai_jobs:
            print(f"AI background removal for {len(ai_jobs)} image(s) with '{ai_model}'...")
            ai_err = run_ai([(str(f), str(png_dir / f"{names[f]}.png")) for f in ai_jobs], ai_model)
            for f in ai_jobs:
                if str(f) in ai_err:
                    errors[f] = f"ai: {ai_err[str(f)]}"
                else:
                    report[f] = ["ai"]
        vec_src = {f: png_dir / f"{names[f]}.png" for f in files if f not in errors}
        print(f"Stage 1 done in {time.time() - t0:.1f}s")

    # ---- Stage 2: vectorize (per-image timeout; retry once in safe mode)
    t1 = time.time()
    from pebble import ProcessPool
    from concurrent.futures import TimeoutError as FutTimeout

    def submit(pool, f, src, safe):
        return pool.schedule(vectorize, args=(str(src), str(svg_dir / f"{names[f]}.svg"),
                             vector_mode, upscale, make_transparent, max_side, safe),
                             timeout=timeout)

    with ProcessPool(max_workers=workers) as pool:
        futs = {submit(pool, f, src, False): (f, src, False) for f, src in vec_src.items()}
        done, total = 0, len(futs)
        while futs:
            for fut in list(futs):
                if not fut.done():
                    continue
                f, src, safe = futs.pop(fut)
                try:
                    report.setdefault(f, []).append(fut.result())
                    done += 1
                    print(f"  ✓ {done}/{total} {names[f]}  [{' → '.join(report[f])}]")
                except FutTimeout:
                    if not safe:
                        print(f"  … {names[f]} too complex, retrying simplified")
                        futs[submit(pool, f, src, True)] = (f, src, True)
                    else:
                        done += 1
                        errors[f] = "vectorize: timed out"
                        print(f"  ✗ {done}/{total} {names[f]}: timed out")
                except Exception as e:
                    done += 1
                    errors[f] = f"vectorize: {e}"
                    print(f"  ✗ {done}/{total} {names[f]}: {e}")
            time.sleep(0.05)
    print(f"Stage 2 done in {time.time() - t1:.1f}s")

    zip_path = shutil.make_archive(str(out), "zip", out)
    ok = len(files) - len(errors)
    print(f"\n✅ {ok}/{len(files)} images converted in {time.time() - t0:.1f}s")
    if errors:
        print("Failed:")
        for f, e in errors.items():
            print(f"  - {f.name}: {e}")
    return zip_path, errors
