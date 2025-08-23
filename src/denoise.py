
import numpy as np
import cv2

def opencv_fastnlmeans(img: np.ndarray,
                       h: float = 7.0,
                       hColor: float = 7.0,
                       templateWindowSize: int = 7,
                       searchWindowSize: int = 21) -> np.ndarray:
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)
    else:
        return cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)

def median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img, ksize)

def bilateral_filter(img: np.ndarray, d: int = 9, sigmaColor: float = 75, sigmaSpace: float = 75) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
