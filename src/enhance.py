# src/enhance.py
from typing import Optional, Tuple
from pathlib import Path
import os
import cv2
import numpy as np

# Gewichts-URLs (offiziell)
WEIGHTS_URLS = {
    "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan-x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "realesrnet-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRNet_x4plus.pth",
    # optional zusätzlich im Code nutzbar:
    "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}

def _ensure_weights(model_name: str, weights: Optional[str]) -> str:
    if weights and Path(weights).exists():
        return weights
    primary = WEIGHTS_URLS.get(model_name)
    if primary is None:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

    mirrors = [primary]
    # kleine Mirror-Liste wenn GitHub blockt
    if model_name == "realesrgan-x4plus":
        mirrors.append("https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth")
    elif model_name == "realesrgan-x2plus":
        mirrors.append("https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2plus.pth")
    elif model_name == "realesrnet-x4plus":
        mirrors.append("https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRNet_x4plus.pth")
    elif model_name == "realesr-general-x4v3":
        mirrors.append("https://huggingface.co/xinntao/Real-ESRGAN/resolve/main/weights/realesr-general-x4v3.pth")

    models_dir = Path("models"); models_dir.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in mirrors:
        dst = models_dir / url.split("/")[-1]
        if dst.exists():
            return str(dst)
        try:
            import requests
            print(f"[INFO] Lade Gewichte: {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
            return str(dst)
        except Exception as e:
            last_err = e
            print(f"[WARN] Download fehlgeschlagen: {e}")
    raise RuntimeError(f"Konnte Gewichte für {model_name} nicht laden: {last_err}")


def _safe_bgr_u8(img: np.ndarray) -> np.ndarray:
    """
    Bringt ein Bild sicher auf BGR uint8, ohne NumPy-Interna (._methods) zu triggern.
    Greift NICHT auf np.max()/np.any() etc. zu, wenn das schiefgehen könnte.
    """
    if img is None:
        raise ValueError("Input image is None")

    # Bereits uint8?
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        return img

    # Falls es kein ndarray ist, defensiv konvertieren
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    # Float-/Complex-Typen?
    if img.dtype.kind in ("f", "c"):
        # Heuristik: konservativ annehmen, dass 0..1 skaliert werden muss.
        # (Das vermeidet max()/any() Aufrufe, die problematische NumPy-Pfade laden.)
        try:
            out = (img * 255.0).astype(np.float32)
        except Exception:
            # Letzter Fallback — minimale Kopie als float32
            out = img.astype(np.float32, copy=False) * 255.0
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        return out

    # Integer-Typen (z.B. uint16) → clampen & auf uint8 mappen
    if img.dtype.kind in ("i", "u"):
        # sanftes clampen ohne .max()
        out = img.astype(np.int32, copy=False)
        out[out < 0] = 0
        out[out > 255] = 255
        return out.astype(np.uint8, copy=False)

    # Andere Fälle: sicherer Fallback
    out = img.astype(np.float32, copy=False)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out

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

def auto_brightness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
