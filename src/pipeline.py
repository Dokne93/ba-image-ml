# src/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import argparse
import sys

from .utils import list_images, read_image, save_image, relative_to_root
from .denoise import opencv_fastnlmeans, median_blur, bilateral_filter
from .enhance import esrgan_superres, unsharp_mask


def process_image(
    img,
    steps: List[str],
    model: str,
    weights: Optional[str] = None,
    tile: int = 0,
    half: bool = False,
):
    """Führt die gewählten Schritte in Reihenfolge aus."""
    out = img
    for s in steps:
        s_l = s.lower()
        if s_l == "denoise":
            out = opencv_fastnlmeans(out)
        elif s_l == "median":
            out = median_blur(out, ksize=3)
        elif s_l == "bilateral":
            out = bilateral_filter(out)
        elif s_l == "esrgan":
            out = esrgan_superres(
                out,
                model_name=model,
                weights=weights,
                tile=tile,
                half=half,
            )
        elif s_l == "sharpen":
            out = unsharp_mask(out, radius=1, amount=0.6)
        else:
            print(f"[WARN] Unknown step: {s}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Eingabeordner")
    ap.add_argument("--output", required=True, help="Ausgabeordner")
    ap.add_argument("--steps", nargs="+", required=True, help="z.B. denoise esrgan sharpen")
    ap.add_argument("--model", default="realesrgan-x4plus", help="Modellname für Super-Resolution")
    ap.add_argument("--weights", default=None, help="Pfad zu Gewichten (optional)")
    ap.add_argument("--tile", type=int, default=0, help="Tile-Größe für ESRGAN (0 = aus, z.B. 128/192/256)")
    ap.add_argument("--half", action="store_true", help="FP16 (nur GPU sinnvoll)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    images = list_images(in_dir)
    if not images:
        print(f"[INFO] Keine Bilder in {in_dir} gefunden.")
        return

    ok, fail = 0, 0
    for p in images:
        try:
            img = read_image(p)
            out = process_image(
                img,
                steps=args.steps,
                model=args.model,
                weights=args.weights,
                tile=args.tile,
                half=args.half,
            )
            rel = relative_to_root(p, in_dir)
            save_image(out_dir / rel, out)
            ok += 1
        except KeyboardInterrupt:
            print("\n[INFO] Abgebrochen.")
            sys.exit(130)
        except Exception as e:
            print(f"[ERR] Verarbeitung fehlgeschlagen für {p}: {e}")
            fail += 1

    print(f"Processed {ok} images (failed: {fail}) -> {out_dir}")


if __name__ == "__main__":
    main()
