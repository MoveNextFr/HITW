import numpy as np
import easyocr

from PIL import Image
from ProtoEqla.lib.prediction import *

GPU = False

class OcrTask(object):
    _instance = None

    def __new__(cls, *args, **kwargs):  # singleton (a reader is expensive in RAM)
        if cls._instance is None:
            cls._instance = super(OcrTask, cls).__new__(cls)
            cls._instance.reader = easyocr.Reader(["fr", "en"], gpu=GPU)
        return cls._instance

    def predict(self, image: Image) -> Prediction:
        raw_results = self.reader.readtext(np.array(image))
        items = []
        id = 0
        for coords, text, score in raw_results:
            bbox = self.__rect_from_coord(coords)
            text = text.replace('\u20ac', '')
            items.append(PredictionItem(text, id, score, bbox))
            id += 1
        return Prediction(image, items)

    def get_text(self, prediction:Prediction):
        items = prediction.items_sorted_on_y_axis_asc()
        return list(map(lambda it: it.label, items))

    @staticmethod
    def __rect_from_coord(coords) -> Rect:
        xs = []
        ys = []
        for x, y in coords:
            xs.append(x)
            ys.append(y)

        return Rect((min(xs), min(ys), max(xs), max(ys)))