# src/enhance.py
from typing import Optional, Tuple
from pathlib import Path
import os
import cv2
import numpy as np

# Gewichts-URLs (offiziell)
WEIGHTS_URLS = {
    "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus.pth",
    "realesrgan-x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x2plus.pth",
    "realesrnet-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRNet_x4plus.pth",
}

def _ensure_weights(model_name: str, weights: Optional[str]) -> str:
    """Sorgt dafür, dass passende Gewichte lokal liegen (./models)."""
    if weights and Path(weights).exists():
        return str(weights)
    url = WEIGHTS_URLS.get(model_name)
    if url is None:
        raise ValueError(f"Unbekanntes Modell: {model_name}")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    dst = models_dir / url.split("/")[-1]
    if not dst.exists():
        import requests
        print(f"[INFO] Lade Gewichte: {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"[INFO] Gewichte gespeichert: {dst}")
    return str(dst)

def _safe_bgr_u8(img: np.ndarray) -> np.ndarray:
    """Sicherstellen: BGR uint8 (RealESRGANer erwartet HWC uint8)."""
    if img is None:
        raise ValueError("Eingabebild ist None.")
    if img.ndim == 2:  # gray -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        if img.max() <= 1.5:  # wahrscheinlich 0..1 float
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    return img

def _auto_tile_for_size(shape: Tuple[int, int, int]) -> int:
    """Heuristik: bei sehr großen Bildern Tiling aktivieren."""
    h, w = shape[:2]
    mpix = (h * w) / 1_000_000.0
    if mpix >= 8:   # ab ~8MP automatisch tilen
        return 256
    return 0

def unsharp_mask(img: np.ndarray, radius: int = 1, amount: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

def esrgan_superres(
    img: np.ndarray,
    model_name: str = "realesrgan-x4plus",
    tile: int = 0,
    half: bool = False,
    denoise_strength: float = 0.0,   # derzeit nicht genutzt; Platzhalter für spätere Varianten
    weights: Optional[str] = None
) -> np.ndarray:
    """
    Super-Resolution mit Real-ESRGAN (stabil via RealESRGANer + RRDBNet).

    - model_name: 'realesrgan-x4plus' | 'realesrgan-x2plus' | 'realesrnet-x4plus'
    - tile: 0 = kein Tiling; >0 z.B. 128/256 für wenig RAM
    - half: nur sinnvoll mit CUDA; auf CPU wird es intern ignoriert
    - weights: Pfad zu *.pth; sonst auto-download in ./models/
    """
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch

        img = _safe_bgr_u8(img)

        scale = 4 if "x4" in model_name else 2
        weights_path = _ensure_weights(model_name, weights)

        # CPU/GPU-Check: half nur auf CUDA sinnvoll
        use_cuda = torch.cuda.is_available()
        if not use_cuda and half:
            print("[INFO] CUDA nicht verfügbar: 'half' wird ignoriert (CPU).")
            half = False

        # Auto-Tiling bei großen Bildern (falls tile nicht gesetzt)
        if tile == 0:
            tile = _auto_tile_for_size(img.shape)
            if tile > 0:
                print(f"[INFO] Auto-Tiling aktiviert: tile={tile}")

        # RRDBNet-Backbone wie in Real-ESRGAN
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32, scale=scale
        )

        upsampler = RealESRGANer(
            scale=scale,
            model_path=weights_path,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=half
        )

        out, _ = upsampler.enhance(img, outscale=scale)
        return out

    except Exception as e:
        print(f"[WARN] Real-ESRGAN fehlgeschlagen: {e}. Rückgabe: Originalbild.")
        return _safe_bgr_u8(img)
