from ProtoEqla.lib.prediction import Prediction
import PIL.Image
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict


def draw_pred(prediction: Prediction, scale_factor: float = 1, font_color="white") -> PIL.Image.Image:
    img = prediction.img_source.copy()
    draw = ImageDraw.Draw(img)
    for item in prediction.items_sorted_on_x_axis_asc():
        x_min, y_min, x_max, y_max = item.bbox.xyxy
        draw.rectangle(item.bbox.xyxy, fill=None, outline="red")
    return img.resize((int(img.width * scale_factor), int(img.height * scale_factor)))