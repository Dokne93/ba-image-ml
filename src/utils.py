
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

def list_images(folder: Path):
    folder = Path(folder)
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in IMG_EXTS],
                  key=lambda p: str(p).lower())

def read_image(path: Path) -> np.ndarray:
    # Mit Pillow laden und EXIF-Orientation anwenden
    with Image.open(str(path)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {'.jpg', '.jpeg'}:
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        cv2.imwrite(str(path), img)

def relative_to_root(path: Path, root: Path) -> Path:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(*path.parts[len(root.parts):])
