
from pathlib import Path
from typing import List, Optional
import argparse
from .utils import list_images, read_image, save_image, relative_to_root
from .denoise import opencv_fastnlmeans, median_blur, bilateral_filter
from .enhance import esrgan_superres, unsharp_mask

def process_image(img, steps: List[str], model: str, weights: Optional[str] = None):
    out = img
    for s in steps:
        if s == 'denoise':
            out = opencv_fastnlmeans(out)
        elif s == 'median':
            out = median_blur(out, ksize=3)
        elif s == 'bilateral':
            out = bilateral_filter(out)
        elif s == 'esrgan':
            out = esrgan_superres(out, model_name=model, weights=weights)
        elif s == 'sharpen':
            out = unsharp_mask(out, radius=1, amount=0.6)
        else:
            print(f"[WARN] Unknown step: {s}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Eingabeordner')
    ap.add_argument('--output', required=True, help='Ausgabeordner')
    ap.add_argument('--steps', nargs='+', required=True, help='z.B. denoise esrgan sharpen')
    ap.add_argument('--model', default='realesrgan-x4plus', help='Modellname für Super-Resolution')
    ap.add_argument('--weights', default=None, help='Pfad zu Gewichten (optional)')
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    images = list_images(in_dir)
    for p in images:
        img = read_image(p)
        out = process_image(img, args.steps, args.model, args.weights)
        rel = relative_to_root(p, in_dir)
        save_image(out_dir / rel, out)

    print(f"Processed {len(images)} images -> {out_dir}")

if __name__ == '__main__':
    main()
