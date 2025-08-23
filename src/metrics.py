# src/metrics.py
from pathlib import Path
import argparse
import csv
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from .utils import list_images, read_image, relative_to_root
import warnings

# ---- Helpers ----

def _ensure_same_size(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resize b to a's size if necessary (bilinear for color, area for downscale)."""
    if a.shape[:2] == b.shape[:2]:
        return b
    warnings.warn(f"Size mismatch: ref={a.shape[:2]}, cmp={b.shape[:2]} -> resizing cmp to ref")
    interp = cv2.INTER_AREA if (b.shape[0] > a.shape[0] or b.shape[1] > a.shape[1]) else cv2.INTER_LINEAR
    return cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=interp)

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure uint8 [0,255] for classical metrics."""
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255) if img.max() > 1.5 else np.clip(img * 255.0, 0, 255)
    return img.astype(np.uint8)

# ---- Metrics ----

def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_uint8(a); b = _to_uint8(b)
    b = _ensure_same_size(a, b)
    return float(psnr(a, b, data_range=255))

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_uint8(a); b = _to_uint8(b)
    b = _ensure_same_size(a, b)
    # SSIM auf Luminanz/Grayscale
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    return float(ssim(a_gray, b_gray, data_range=255))

# LPIPS einmalig lazy-laden (spart Zeit)
_lpips_fn = None
def _get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='vgg')
    return _lpips_fn

def compute_lpips(a: np.ndarray, b: np.ndarray) -> float:
    try:
        import torch
        a = _to_uint8(a); b = _to_uint8(b)
        b = _ensure_same_size(a, b)

        # HWC -> 3-kanalig RGB (LPIPS erwartet RGB, float32 in [-1,1])
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        if b.ndim == 2:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

        a_rgb = np.ascontiguousarray(a[:, :, ::-1])  # BGR->RGB, kontigu (fix negative strides)
        b_rgb = np.ascontiguousarray(b[:, :, ::-1])

        a_t = torch.from_numpy(a_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
        b_t = torch.from_numpy(b_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1

        loss_fn = _get_lpips()
        with torch.no_grad():
            d = loss_fn(a_t, b_t).item()
        return float(d)
    except Exception as e:
        warnings.warn(f"LPIPS not available: {e}")
        return float('nan')

# ---- CLI ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='Pfad zu Referenzbildern (Originale)')
    ap.add_argument('--cmp', required=True, help='Pfad zu Vergleichsbildern (verarbeitet)')
    ap.add_argument('--metrics', nargs='+', default=['psnr', 'ssim', 'lpips'],
                    help='Welche Metriken berechnen (psnr ssim lpips)')
    ap.add_argument('--out', default='outputs/quality.csv', help='CSV-Ausgabe')
    args = ap.parse_args()

    ref_dir = Path(args.ref)
    cmp_dir = Path(args.cmp)
    ref_imgs = list_images(ref_dir)

    rows = []
    for ref in ref_imgs:
        rel = relative_to_root(ref, ref_dir)
        cmp_path = cmp_dir / rel
        if not cmp_path.exists():
            warnings.warn(f"Vergleichsbild fehlt, übersprungen: {cmp_path}")
            continue

        a = read_image(ref)
        b = read_image(cmp_path)

        result = {'image': str(rel)}
        if 'psnr' in args.metrics:
            result['psnr'] = compute_psnr(a, b)
        if 'ssim' in args.metrics:
            result['ssim'] = compute_ssim(a, b)
        if 'lpips' in args.metrics:
            result['lpips'] = compute_lpips(a, b)

        rows.append(result)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        fieldnames = rows[0].keys() if rows else ['image']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == '__main__':
    main()
