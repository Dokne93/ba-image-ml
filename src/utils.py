
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

def list_images(folder: Path):
    return sorted([p for p in Path(folder).rglob('*') if p.suffix.lower() in IMG_EXTS])

def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def relative_to_root(path: Path, root: Path) -> Path:
    return Path(*Path(path).parts[len(Path(root).parts):])
