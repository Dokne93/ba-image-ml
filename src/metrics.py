from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
import warnings
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.utils import read_image  # EXIF-aware

import cv2
import numpy as np

# Unterdrücke UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---- optionale Libs: Versuchen und speichern, was verfügbar ist
HAS_TQDM = True
try:
    from tqdm import tqdm
except Exception:
    HAS_TQDM = False

HAS_SKIMG = True
try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import niqe as sk_niqe  # no-reference
except Exception:
    HAS_SKIMG = False

HAS_SEWAR_FSIM = True
HAS_SEWAR_VIF = True
try:
    from sewar.full_ref import fsim as sewar_fsim  # fehlt in manchen Versionen
except Exception:
    HAS_SEWAR_FSIM = False
try:
    from sewar.full_ref import vifp as sewar_vif  # VIF (pixel domain)
except Exception:
    HAS_SEWAR_VIF = False

# PIQ (bevorzugt für LPIPS & DISTS, vermeidet torchvision)
HAS_PIQ = True
try:
    import torch
    import piq  # DISTS, FSIM-Fallback, LPIPS
except Exception:
    HAS_PIQ = False

# ---- BRISQUE lazy load (verhindert Import-Crash, wenn 'svmutil' fehlt)
_HAS_BRISQUE = None
_BRISQUE_MODEL = None

def _ensure_brisque_loaded():
    global _HAS_BRISQUE, _BRISQUE_MODEL
    if _HAS_BRISQUE is not None:
        return _HAS_BRISQUE
    try:
        from brisque import BRISQUE  # benötigt 'svmutil' (libsvm-official)
        _BRISQUE_MODEL = BRISQUE()
        _HAS_BRISQUE = True
    except Exception:
        _BRISQUE_MODEL = None
        _HAS_BRISQUE = False
    return _HAS_BRISQUE


# LPIPS-Paket NICHT hier importieren (zieht torchvision & produziert Warnungen).
# Lazy-Import nur wenn LPIPS wirklich angefordert UND PIQ nicht verfügbar ist.
_HAS_LPIPS_PKG = None
def _lazy_import_lpips():
    global _HAS_LPIPS_PKG
    if _HAS_LPIPS_PKG is None:
        try:
            import lpips  # type: ignore
            _HAS_LPIPS_PKG = lpips
        except Exception:
            _HAS_LPIPS_PKG = False
    return _HAS_LPIPS_PKG if _HAS_LPIPS_PKG is not False else None


# ---------------------------
# Utils
# ---------------------------
def _set_thread_limits(n: int):
    # Begrenze BLAS/NumPy/Torch Threads pro Prozess (wichtig bei MP)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    try:
        import torch as _torch  # noqa
        if hasattr(_torch, "set_num_threads"):
            _torch.set_num_threads(max(1, int(n)))
    except Exception:
        pass
    try:
        cv2.setNumThreads(max(1, int(n)))
    except Exception:
        pass


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _ensure_same_size(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape[:2] == b.shape[:2]


# ---------------------------
# Metriken
# ---------------------------
def compute_psnr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        if HAS_SKIMG:
            return float(sk_psnr(a, b, data_range=255))
        mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
        if mse <= 1e-12:
            return float("inf")
        return float(10.0 * np.log10((255.0 * 255.0) / mse))
    except Exception:
        return None


def compute_ssim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    SSIM in reiner NumPy/OpenCV-Implementierung (kein scikit-image nötig).
    - Rechnet auf der Y-Luma (BT.601) für Farbbilder.
    - Fenster: Gauß 11x11, sigma=1.5
    """
    try:
        if a.shape != b.shape:
            return None

        # Falls Farbe: auf Y (Luma) wechseln – stabil
        if a.ndim == 3 and a.shape[2] == 3:
            # BGR -> Y (0..255, float32)
            a_y = cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
            b_y = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
        else:
            a_y = a.astype(np.float32)
            b_y = b.astype(np.float32)

        # Gauß-Fenster
        ksize = (11, 11)
        sigma = 1.5
        mu1 = cv2.GaussianBlur(a_y, ksize, sigma)
        mu2 = cv2.GaussianBlur(b_y, ksize, sigma)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(a_y * a_y, ksize, sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(b_y * b_y, ksize, sigma) - mu2_sq
        sigma12   = cv2.GaussianBlur(a_y * b_y, ksize, sigma) - mu1_mu2

        # SSIM Konstanten (L = 255)
        L = 255.0
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = num / (den + 1e-12)
        return float(ssim_map.mean())
    except Exception:
        return None




def compute_mse(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    except Exception:
        return None


def compute_mae(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    except Exception:
        return None


def compute_vif(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        if HAS_SEWAR_VIF:
            ag = _to_gray(a); bg = _to_gray(b)
            return float(sewar_vif(ag, bg))
        return None
    except Exception:
        return None


def compute_fsim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    # erst sewar, dann piq als Fallback
    try:
        if HAS_SEWAR_FSIM:
            ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
            bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
            return float(sewar_fsim(ag, bg))
    except Exception:
        pass

    if HAS_PIQ:
        try:
            at = torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            bt = torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            val = piq.fsim(at, bt, data_range=1.0)
            return float(val.detach().cpu().item())
        except Exception:
            return None
    return None


# LPIPS: bevorzugt PIQ (vermeidet torchvision). Fallback: lpips-Paket.
_LPIPS_MODEL_PIQ = None
_LPIPS_MODEL_LPIPS = None

def _get_piq_lpips():
    global _LPIPS_MODEL_PIQ
    if not HAS_PIQ:
        return None
    if _LPIPS_MODEL_PIQ is None:
        try:
            _LPIPS_MODEL_PIQ = piq.LPIPS(net='alex', reduction='none')  # klein & CPU-freundlich
        except Exception:
            _LPIPS_MODEL_PIQ = False
    return _LPIPS_MODEL_PIQ if _LPIPS_MODEL_PIQ is not False else None


def _get_lpips_pkg_model():
    global _LPIPS_MODEL_LPIPS
    if _LPIPS_MODEL_LPIPS is None:
        lpips_pkg = _lazy_import_lpips()
        if lpips_pkg is None:
            _LPIPS_MODEL_LPIPS = False
        else:
            try:
                _LPIPS_MODEL_LPIPS = lpips_pkg.LPIPS(net='alex')
            except Exception:
                _LPIPS_MODEL_LPIPS = False
    return _LPIPS_MODEL_LPIPS if _LPIPS_MODEL_LPIPS is not False else None


def compute_lpips(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    # Erst PIQ-Implementierung, ansonsten Fallback auf lpips-Paket
    try:
        model = _get_piq_lpips()
        if model is not None:
            a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            at = torch.from_numpy(a_rgb).permute(2, 0, 1).unsqueeze(0)
            bt = torch.from_numpy(b_rgb).permute(2, 0, 1).unsqueeze(0)
            val = model(at, bt)  # Tensor [1]
            return float(val.mean().item())
    except Exception:
        pass

    try:
        model = _get_lpips_pkg_model()
        if model is None:
            return None
        a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        at = torch.from_numpy(a_rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        bt = torch.from_numpy(b_rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        val = model(at, bt)
        return float(val.detach().cpu().numpy().item())
    except Exception:
        return None


def compute_dists(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not HAS_PIQ:
        return None
    try:
        a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        at = torch.from_numpy(a_rgb).permute(2, 0, 1).unsqueeze(0)
        bt = torch.from_numpy(b_rgb).permute(2, 0, 1).unsqueeze(0)
        model = piq.DISTS(reduction='none')
        val = model(at, bt)
        return float(val.mean().item())
    except Exception:
        return None


def compute_brisque(img: np.ndarray) -> Optional[float]:
    if not _ensure_brisque_loaded():
        return None
    try:
        return float(_BRISQUE_MODEL.get_score(img))
    except Exception:
        return None




def compute_niqe(img: np.ndarray) -> Optional[float]:
    if not HAS_SKIMG:
        return None
    try:
        g = img
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        g = g.astype(np.float32) / 255.0
        return float(sk_niqe(g))
    except Exception:
        return None


# ---------------------------
# Worker
# ---------------------------
@dataclass
class Job:
    ref_path: Path
    cmp_path: Path
    metrics: List[str]


def _process_one(job: Job, threads: int) -> Tuple[str, Dict[str, Optional[float]], Optional[str]]:
    """Gibt (image_name, results_dict, error_str|None) zurück."""
    _set_thread_limits(max(1, int(threads)))

    ref, cmp = job.ref_path, job.cmp_path
    try:
        # a = cv2.imread(str(ref), cv2.IMREAD_UNCHANGED)
        # b = cv2.imread(str(cmp), cv2.IMREAD_UNCHANGED)
        a = read_image(ref)
        b = read_image(cmp)
        if a is None or b is None:
            return (ref.name, {}, f"Unreadable image(s): {ref} / {cmp}")
        if not _ensure_same_size(a, b):
            return (ref.name, {}, f"Size mismatch: {a.shape} vs {b.shape}")

        res: Dict[str, Optional[float]] = {}
        # Referenzierte Metriken
        if 'psnr' in job.metrics:
            res['psnr'] = compute_psnr(a, b)
        if 'ssim' in job.metrics:
            res['ssim'] = compute_ssim(a, b)
        if 'lpips' in job.metrics:
            res['lpips'] = compute_lpips(a, b)
        if 'mse' in job.metrics:
            res['mse'] = compute_mse(a, b)
        if 'mae' in job.metrics:
            res['mae'] = compute_mae(a, b)
        if 'fsim' in job.metrics:
            res['fsim'] = compute_fsim(a, b)
        if 'vif' in job.metrics:
            res['vif'] = compute_vif(a, b)
        # No-reference (nur auf cmp)
        if 'brisque' in job.metrics:
            res['brisque_cmp'] = compute_brisque(b)
        if 'niqe' in job.metrics:
            res['niqe_cmp'] = compute_niqe(b)
        if 'dists' in job.metrics:
            res['dists'] = compute_dists(a, b)

        return (str(ref.name), res, None)
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return (str(ref.name), {}, f"{e}\n{tb}")


# ---------------------------
# CLI
# ---------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Qualitätsmetriken zwischen Referenz- und Vergleichsbildern berechnen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ref", required=True, help="Ordner mit Referenzbildern")
    ap.add_argument("--cmp", required=True, help="Ordner mit Vergleichsbildern")
    ap.add_argument("--metrics", nargs="+", required=True,
                    help="z. B. psnr ssim lpips mse mae fsim vif brisque niqe dists")
    ap.add_argument("--out", required=True, help="CSV-Ausgabedatei")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2),
                    help="Parallelprozesse")
    ap.add_argument("--threads", type=int, default=1,
                    help="CPU-Threads pro Prozess (Torch/OpenCV/BLAS)")
    ap.add_argument("--fail-fast", action="store_true", help="Bei erstem Fehler abbrechen")
    ap.add_argument("--verbose", "-v", action="count", default=0, help="Mehr Ausgaben")
    return ap


def main():
    args = _build_parser().parse_args()

    ref_dir = Path(args.ref)
    cmp_dir = Path(args.cmp)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Bilde Paare über identische Dateinamen (einschließlich Endung)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    refs = sorted([p for p in ref_dir.iterdir() if p.suffix.lower() in exts], key=lambda p: p.name.lower())

    jobs: List[Job] = []
    missing = []
    for rp in refs:
        cand = cmp_dir / rp.name
        if not cand.exists():
            missing.append(rp.name)
            continue
        jobs.append(Job(ref_path=rp, cmp_path=cand, metrics=[m.lower() for m in args.metrics]))

    if args.verbose:
        print(f"Gefundene Referenzen: {len(refs)} | Matched Paare: {len(jobs)} | Missing: {len(missing)}")
        if missing and args.verbose >= 2:
            print("Fehlend (Beispiele):", missing[:10])

    # Multiprocessing
    results: List[Tuple[str, Dict[str, Optional[float]], Optional[str]]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, int(args.workers))) as pool:
        it = (pool.apply_async(_process_one, (job, args.threads)) for job in jobs)

        if HAS_TQDM:
            for i, fut in enumerate(tqdm(it, total=len(jobs), desc="Berechne Metriken", unit="img")):
                r = fut.get()
                results.append(r)
                if args.fail_fast and r[2] is not None:
                    print("Abbruch wegen Fehler:", r[2], file=sys.stderr)
                    break
        else:
            for i, fut in enumerate(it):
                r = fut.get()
                print(f"[{i+1}/{len(jobs)}] {r[0]} {'OK' if r[2] is None else 'ERR'}")
                results.append(r)
                if args.fail_fast and r[2] is not None:
                    print("Abbruch wegen Fehler:", r[2], file=sys.stderr)
                    break

    # CSV schreiben – Spalten dynamisch je nach angeforderten Metriken
    header = ["image"]
    for m in [m.lower() for m in args.metrics]:
        if m in ("brisque", "niqe"):
            col = "brisque_cmp" if m == "brisque" else "niqe_cmp"
            if col not in header:
                header.append(col)
        else:
            if m not in header:
                header.append(m)
    header.append("error")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, res, err in results:
            row = [name]
            for col in header[1:-1]:
                row.append(res.get(col))
            row.append(err)
            writer.writerow(row)

    print(f"Fertig -> {out_csv} (Paare: {len(jobs)}, Fehler: {sum(1 for _,_,e in results if e)})")


if __name__ == "__main__":
    # global 'spawn' ist stabiler
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
