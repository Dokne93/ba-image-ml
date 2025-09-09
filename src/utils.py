from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

def list_images(folder: Path):
    return sorted([p for p in Path(folder).rglob('*') if p.suffix.lower() in IMG_EXTS])

def read_image(path: Path):
    # EXIF-Orientation automatisch anwenden
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # macht aus Flag echte Rotation
        im = im.convert("RGB")
        arr = np.array(im)[:, :, ::-1]  # RGB -> BGR für OpenCV
    return arr

def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Schreibe JPEG mit hoher Qualität, ohne EXIF-Orientation-Flags
    if path.suffix.lower() in ('.jpg', '.jpeg'):
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    else:
        cv2.imwrite(str(path), img)

def relative_to_root(path: Path, root: Path) -> Path:
    return Path(*Path(path).parts[len(Path(root).parts):])
