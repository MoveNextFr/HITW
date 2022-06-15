import cv2
import numpy as np


def histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist


def is_correctly_exposed(image, threshold=0.5) -> bool:
    hist = histogram(image)
    return np.sum(hist[225:]) / np.sum(hist) <= threshold and np.sum(hist[0:25]) / np.sum(hist) <= threshold


def image_info(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    correct_expo = is_correctly_exposed(image)
    return height, width, (width, height), correct_expo
