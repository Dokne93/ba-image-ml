# src/metrics.py
from pathlib import Path
import argparse
import csv
import warnings
from typing import Dict

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from .utils import list_images, read_image, relative_to_root

# =========================
# Helpers
# =========================

def _ensure_same_size(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resize b to a's size if necessary (bilinear for up, area for down)."""
    if a.shape[:2] == b.shape[:2]:
        return b
    warnings.warn(f"Size mismatch: ref={a.shape[:2]}, cmp={b.shape[:2]} -> resizing cmp to ref")
    # Wenn b größer als a, dann downscale mit AREA (besser), sonst LINEAR
    if b.shape[0] > a.shape[0] or b.shape[1] > a.shape[1]:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=interp)

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure uint8 [0,255] for classical metrics."""
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.5:
        img = img * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# =========================
# PSNR / SSIM / LPIPS
# =========================

def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_uint8(a); b = _to_uint8(_ensure_same_size(a, b))
    return float(psnr(a, b, data_range=255))

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_uint8(a); b = _to_uint8(_ensure_same_size(a, b))
    # SSIM auf Luminanz/Grayscale
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    return float(ssim(a_gray, b_gray, data_range=255))

# LPIPS einmalig lazy-laden (spart Zeit)
_lpips_fn = None
def _get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips, torch
        _lpips_fn = lpips.LPIPS(net='vgg').to('cpu')
    return _lpips_fn

def compute_lpips(a: np.ndarray, b: np.ndarray) -> float:
    try:
        import torch
        a = _to_uint8(a); b = _to_uint8(_ensure_same_size(a, b))

        # HWC -> 3-kanalig RGB (LPIPS erwartet RGB, float32 in [-1,1])
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        if b.ndim == 2:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

        a_rgb = np.ascontiguousarray(a[:, :, ::-1])  # BGR->RGB
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

# =========================
# Zusätzliche Metriken
# =========================

def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_uint8(a); b = _to_uint8(_ensure_same_size(a, b))
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_uint8(a); b = _to_uint8(_ensure_same_size(a, b))
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

def compute_fsim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from sewar.full_ref import fsim
        a = _to_uint8(a); b = _to_uint8(_ensure_same_size(a, b))
        a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if a.ndim == 3 else cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB) if b.ndim == 3 else cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
        val = fsim(a_rgb, b_rgb)
        return float(val[0] if isinstance(val, (tuple, list)) else val)
    except Exception as e:
        warnings.warn(f"FSIM nicht verfügbar: {e}")
        return float('nan')

def compute_vif(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from sewar.full_ref import vifp
        a = _to_uint8(a); b = _to_uint8(_ensure_same_size(a, b))
        a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
        val = vifp(a_gray, b_gray)
        return float(val[0] if isinstance(val, (tuple, list)) else val)
    except Exception as e:
        warnings.warn(f"VIF nicht verfügbar: {e}")
        return float('nan')

# pyiqa-Metriken cachen (BRISQUE/NIQE/DISTS), damit nicht pro Bild neu geladen wird
_pyiqa_cache: Dict[str, object] = {}
def _get_pyiqa(name: str):
    try:
        import pyiqa
        if name not in _pyiqa_cache:
            _pyiqa_cache[name] = pyiqa.create_metric(name, device='cpu')
        return _pyiqa_cache[name]
    except Exception as e:
        raise RuntimeError(f"pyiqa Metric '{name}' nicht verfügbar: {e}")

def compute_brisque(a: np.ndarray) -> float:
    """
    No-reference Qualität (niedriger = besser) per BRISQUE (pyiqa).
    Erwartet OpenCV-Input (BGR uint8).
    """
    try:
        import torch
        a = _to_uint8(a)
        # BGR -> RGB, [0,1], NCHW
        if a.ndim == 2:
            a_rgb = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        else:
            a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(a_rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        metric = _get_pyiqa('brisque')
        with torch.no_grad():
            score = metric(t).item()
        return float(score)
    except Exception as e:
        warnings.warn(f"BRISQUE nicht verfügbar: {e}")
        return float('nan')

def compute_niqe(a: np.ndarray) -> float:
    """No-reference Qualität (niedriger = besser) per NIQE (pyiqa)."""
    try:
        import torch
        a = _to_uint8(a)
        if a.ndim == 2:
            a_rgb = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        else:
            a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(a_rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        metric = _get_pyiqa('niqe')
        with torch.no_grad():
            score = metric(t).item()
        return float(score)
    except Exception as e:
        warnings.warn(f"NIQE nicht verfügbar: {e}")
        return float('nan')

def compute_dists(a: np.ndarray, b: np.ndarray) -> float:
    """Referenzbasierte Qualität (niedriger = besser) per DISTS (pyiqa)."""
    try:
        import torch
        a = _to_uint8(a); b = _to_uint8(_ensure_same_size(a, b))
        a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if a.ndim == 3 else cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB) if b.ndim == 3 else cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
        ta = torch.from_numpy(a_rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        tb = torch.from_numpy(b_rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        metric = _get_pyiqa('dists')
        with torch.no_grad():
            score = metric(ta, tb).item()
        return float(score)
    except Exception as e:
        warnings.warn(f"DISTS nicht verfügbar: {e}")
        return float('nan')

# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='Pfad zu Referenzbildern (Originale)')
    ap.add_argument('--cmp', required=True, help='Pfad zu Vergleichsbildern (verarbeitet)')
    ap.add_argument('--metrics', nargs='+',
                    default=['psnr', 'ssim', 'lpips'],
                    help='Welche Metriken berechnen (z.B. psnr ssim lpips mse mae fsim vif brisque niqe dists)')
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
        if 'mse' in args.metrics:
            result['mse'] = compute_mse(a, b)
        if 'mae' in args.metrics:
            result['mae'] = compute_mae(a, b)
        if 'fsim' in args.metrics:
            result['fsim'] = compute_fsim(a, b)
        if 'vif' in args.metrics:
            result['vif'] = compute_vif(a, b)
        # No-reference Metriken nur auf dem Vergleichsbild (b)
        if 'brisque' in args.metrics:
            result['brisque_cmp'] = compute_brisque(b)
        if 'niqe' in args.metrics:
            result['niqe_cmp'] = compute_niqe(b)
        if 'dists' in args.metrics:
            result['dists'] = compute_dists(a, b)

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
