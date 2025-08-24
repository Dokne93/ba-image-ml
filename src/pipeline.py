# src/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import argparse
import cv2

from .utils import list_images, read_image, save_image, relative_to_root
from .denoise import opencv_fastnlmeans, median_blur, bilateral_filter
from .enhance import esrgan_superres, unsharp_mask


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
        else:
            print(f"[WARN] Unknown step: {s}")
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Eingabeordner")
    ap.add_argument("--output", required=True, help="Ausgabeordner")
    ap.add_argument("--steps", nargs="+", required=True, help="z.B. denoise esrgan sharpen")
    ap.add_argument("--model", default="realesrgan-x4plus", help="Modell für Super-Resolution")
    ap.add_argument("--weights", default=None, help="Pfad zu Gewichten (optional)")
    ap.add_argument("--tile", type=int, default=0, help="Tile-Größe für ESRGAN (0=aus, z.B. 256)")
    ap.add_argument("--half", action="store_true", help="FP16 (GPU sinnvoll), CPU wird i.d.R. ignoriert")
    ap.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Vor ESRGAN lange Kante auf <= max-side skalieren (0=aus)",
    )
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
    ok, fail = 0, 0

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

    print(f"Processed {ok} images (failed: {fail}) -> {out_dir}")
