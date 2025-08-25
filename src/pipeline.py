# src/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import argparse
import os
import gc
import traceback

import cv2
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# interne Imports
from .utils import list_images, read_image, save_image, relative_to_root
from .denoise import opencv_fastnlmeans, median_blur, bilateral_filter
from .enhance import esrgan_superres, unsharp_mask, auto_brightness


# ---------------------------
# Thread-/BLAS-Limits je Prozess
# ---------------------------
def _set_thread_limits(n: int):
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n)
    try:
        import torch
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(n)
    except Exception:
        pass
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass


# ---------------------------
# Worker-Funktion für Parallelbetrieb
# ---------------------------
def _process_one(
    p_str: str,
    in_dir: str,
    out_dir: str,
    steps: List[str],
    model: str,
    weights: Optional[str],
    tile: int,
    half: bool,
    max_side: int,
    threads: int,
):
    """Wird in Subprozessen ausgeführt – gibt (pfad, fehler|None) zurück."""
    from pathlib import Path as _Path
    _set_thread_limits(threads)

    try:
        p = _Path(p_str)
        in_dir_p = _Path(in_dir)
        out_dir_p = _Path(out_dir)

        img = read_image(p)
        out = process_image(
            img,
            steps=steps,
            model=model,
            weights=weights,
            tile=tile,
            half=half,
            max_side=max_side,
        )
        rel = relative_to_root(p, in_dir_p)
        save_image(out_dir_p / rel, out)
        return (p_str, None)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return (p_str, f"{e}\n{tb}")
    finally:
        try:
            del img, out
        except Exception:
            pass
        gc.collect()


# ---------------------------
# Hilfsfunktionen
# ---------------------------
def _resize_long_side(img, max_side: int):
    """Skaliert so, dass die lange Kante <= max_side (nur wenn nötig)."""
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def process_image(
    img,
    steps: List[str],
    model: str,
    weights: Optional[str] = None,
    tile: int = 0,
    half: bool = False,
    max_side: int = 0,
):
    out = img
    for s in steps:
        if s == "denoise":
            out = opencv_fastnlmeans(out)
        elif s == "median":
            out = median_blur(out, ksize=3)
        elif s == "bilateral":
            out = bilateral_filter(out)
        elif s == "esrgan":
            # optionales Vor-Scaling, um Speicher/Runtime zu zähmen
            out = _resize_long_side(out, max_side)
            out = esrgan_superres(
                out,
                model_name=model,
                weights=weights,
                tile=tile,
                half=half,
            )
        elif s == "sharpen":
            out = unsharp_mask(out, radius=1, amount=0.6)
        elif s == "autobright":
            out = auto_brightness(out)
        else:
            print(f"[WARN] Unknown step: {s}")
    return out


# ---------------------------
# CLI
# ---------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Eingabeordner")
    ap.add_argument("--output", required=True, help="Ausgabeordner")
    ap.add_argument("--steps", nargs="+", required=True, help="z. B. denoise esrgan sharpen autobright")
    ap.add_argument("--model", default="realesrgan-x4plus", help="Modell für Super-Resolution")
    ap.add_argument("--weights", default=None, help="Pfad zu Gewichten (optional)")

    # ESRGAN-Optionen
    ap.add_argument("--tile", type=int, default=0, help="Tile-Größe für ESRGAN (0=aus, z. B. 256)")
    ap.add_argument("--half", action="store_true", help="FP16 (nur sinnvoll mit GPU); CPU wird meist ignoriert")
    ap.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Vor ESRGAN lange Kante auf <= max-side skalieren (0=aus)",
    )

    # Parallelisierung
    ap.add_argument("--workers", type=int, default=1, help="Anzahl Prozesse (1=seriell)")
    ap.add_argument("--threads", type=int, default=1, help="CPU-Threads je Prozess")
    return ap


def main(args: Optional[argparse.Namespace] = None):
    # args entweder von cli.py übergeben oder hier parsen
    if args is None:
        ap = _build_parser()
        args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(in_dir)

    workers = int(getattr(args, "workers", 1))
    threads = int(getattr(args, "threads", 1))

    ok, fail = 0, 0

    if workers <= 1:
        # -------- Serieller Modus (robust, speicherschonend) --------
        _set_thread_limits(threads)
        for p in images:
            try:
                img = read_image(p)
                out = process_image(
                    img,
                    steps=args.steps,
                    model=args.model,
                    weights=args.weights,
                    tile=getattr(args, "tile", 0),
                    half=getattr(args, "half", False),
                    max_side=getattr(args, "max_side", 0),
                )
                rel = relative_to_root(p, in_dir)
                save_image(out_dir / rel, out)
                ok += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[ERROR] {p}: {e}")
                fail += 1
            finally:
                try:
                    del img, out
                except Exception:
                    pass
                gc.collect()
    else:
        # -------- Parallel (mehrere Prozesse) --------
        func = partial(
            _process_one,
            in_dir=str(in_dir),
            out_dir=str(out_dir),
            steps=args.steps,
            model=args.model,
            weights=args.weights,
            tile=getattr(args, "tile", 0),
            half=getattr(args, "half", False),
            max_side=getattr(args, "max_side", 0),
            threads=threads,
        )
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(func, str(p)): p for p in images}
            for fut in as_completed(futures):
                path_str, err = fut.result()
                if err is None:
                    ok += 1
                else:
                    print(f"[ERROR] {path_str}: {err.strip()}")
                    fail += 1

    print(f"Processed {ok} images (failed: {fail}) -> {out_dir}")


if __name__ == "__main__":
    main()
