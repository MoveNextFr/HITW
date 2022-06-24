from dataclasses import dataclass
from typing import List, Optional, Tuple

import PIL.Image
import math
from PIL import Image


@dataclass
class Rect:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __init__(self, xyxy: Tuple[float, float, float, float]):
        self.x_min, self.y_min, self.x_max, self.y_max = xyxy

        if self.x_max < self.x_min:
            inversed_x_max = self.x_min
            inversed_x_min = self.x_max
            self.x_max = inversed_x_max
            self.x_min = inversed_x_min

        if self.y_max < self.y_min:
            inversed_y_max = self.y_min
            inversed_y_min = self.y_max
            self.y_max = inversed_y_max
            self.y_min = inversed_y_min

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def center_x(self) -> float:
        return self.x_min + (self.x_max - self.x_min) / 2

    @property
    def center_y(self) -> float:
        return self.y_min + (self.y_max - self.y_min) / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersect(self, other: "Rect") -> Optional["Rect"]:
        x_min = max(self.x_min, other.x_min)
        x_max = min(self.x_max, other.x_max)
        y_min = max(self.y_min, other.y_min)
        y_max = min(self.y_max, other.y_max)
        if x_min <= x_max and y_min <= y_max:
            return Rect((x_min, y_min, x_max, y_max))
        else:
            return None

    def compute_iou(self, other: "Rect") -> float:
        intersect = self.intersect(other)
        if intersect is None:
            return 0

        intersection_area = intersect.area
        union_area = self.area + other.area - intersection_area

        if union_area == 0:
            return 0

        return intersection_area / union_area

    def center_euclidian_distance(self, other: "Rect") -> float:
        return math.sqrt((self.center_x - other.center_x) ** 2 + (self.center_y - other.center_y) ** 2)

    def contains(self, x, y) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def enlarge(self, x_scale=10, y_scale=10):
        ratio_width = (x_scale * self.width) / 100
        ratio_height = (y_scale * self.height) / 100
        self.x_min = self.x_min - ratio_width
        self.y_min = self.y_min - ratio_height
        self.x_max = self.x_max + ratio_width
        self.y_max = self.y_max + ratio_height


@dataclass
class PredictionItem:
    label: str
    label_id: int
    score: float
    bbox: Optional[Rect]

    def iou(self, other: "PredictionItem") -> float:
        return self.bbox.compute_iou(other.bbox)

    def set_label(self, new_label: str):
        self.label = new_label


@dataclass
class Prediction:
    img_source: Image
    items: List[PredictionItem]

    def __init__(self, img_source: PIL.Image.Image, items: List[PredictionItem]):
        self.img_source = img_source
        self.items = Prediction.__get_items_sorted_by_score_desc(items)

    @staticmethod
    def __get_items_sorted_by_score_desc(items: List[PredictionItem]) -> List[PredictionItem]:
        return list(sorted(items, key=lambda it: it.score, reverse=True))

    def img_source_size(self) -> (float, float):
        return self.img_source.width, self.img_source.height

    def get_item(self, index: int) -> PredictionItem:
        return self.items[index]

    def remove_item_at(self, index: int) -> bool:
        raise NotImplementedError()

    def get_bboxes(self) -> List[Rect]:
        return list(map(lambda it: it.bbox, self.items))

    def get_bboxes_xyxy(self) -> List[Tuple[float, float, float, float]]:
        return list(map(lambda it: it.bbox.xyxy, self.items))

    def get_scores(self) -> List[float]:
        return list(map(lambda it: it.score, self.items))

    def with_nms(self, iou_threshold: float) -> "Prediction":
        prediction_items = self.items
        if len(prediction_items) <= 1:
            return self

        nms_items = []
        for pi in prediction_items:
            is_selected = True
            for pj in prediction_items:
                if pi.iou(pj) > iou_threshold:
                    if pj.score > pi.score:
                        is_selected = False
            if is_selected:
                nms_items.append(pi)
        return Prediction(self.img_source, nms_items)

    def with_best_scores(self) -> "Prediction":
        labels = set(map(lambda it: it.label, self.items))
        best_items = list(map(lambda it: self.item_with_best_score(it), labels))
        return Prediction(self.img_source, best_items)

    def with_grouped_items(self, base_threshold_x: int = 10, base_threshold_y: int = 10, base_image_width: int = 0,
                           base_image_height: int = 0, grp_predicate=None, single_group_labels: List[str] = None) -> "Prediction":
        if len(self.items) < 2:
            return self

        img_source_width, img_source_height = self.img_source_size()
        threshold_x = (base_threshold_x * img_source_width) / base_image_width
        threshold_y = (base_threshold_y * img_source_height) / base_image_height

        items = self.items_sorted_on_x_then_on_y_axis_asc()

        single_group_labels = single_group_labels if single_group_labels is not None else []
        single_group_items = list(filter(lambda it: it.label in single_group_labels, items))
        items = list(filter(lambda it: it.label not in single_group_labels, items))

        current_group = Grp(threshold_x, threshold_y, [items[0]])
        groups = [current_group]
        for item in items[1:]:
            g = next(filter(lambda it: it.accept(item), groups), None)
            if g is None:
                g = Grp(threshold_x, threshold_y, [item])
                groups.append(g)
            else:
                g.append(item)

        new_items = list(map(lambda it: it.to_pred_item(grp_predicate), groups)) + single_group_items
        new_items = Prediction(self.img_source, new_items).items_sorted_on_x_then_on_y_axis_asc()
        return Prediction(self.img_source, new_items)

    def keep_item_with_best_score(self, label: str) -> "Prediction":
        best_score_item = self.item_with_best_score(label)
        items = list(filter(lambda it: it.label != label, self.items))
        items.append(best_score_item)
        return Prediction(self.img_source, items)

    def single_item_with_best_score(self) -> Optional[PredictionItem]:
        sorted_list = sorted(self.items, key=lambda it: it.score, reverse=True)
        return sorted_list[0] if len(sorted_list) > 0 else None

    def item_with_best_score(self, label: str) -> Optional[PredictionItem]:
        return next(filter(lambda it: it.label == label, self.items), None)

    def replace_label(self, old_label: str, new_label: str):
        for item in self.items:
            if item.label == old_label:
                item.set_label(new_label)

    def get_labels(self) -> List[str]:
        return list(map(lambda it: it.score, self.items))

    def items_sorted_on_x_axis_asc(self) -> List[PredictionItem]:
        return list(sorted(self.items, key=lambda it: it.bbox.x_min, reverse=False))

    def items_sorted_on_y_axis_asc(self) -> List[PredictionItem]:
        return list(sorted(self.items, key=lambda it: it.bbox.center_y, reverse=False))

    def items_sorted_on_y_axis_desc(self) -> List[PredictionItem]:
        return list(sorted(self.items, key=lambda it: it.bbox.center_y, reverse=True))

    def items_sorted_on_y_then_on_x_axis_asc(self) -> List[PredictionItem]:
        return list(sorted(self.items, key=lambda it: (it.bbox.center_y, it.bbox.center_x), reverse=False))

    def items_sorted_on_x_then_on_y_axis_asc(self) -> List[PredictionItem]:
        return list(sorted(self.items, key=lambda it: (it.bbox.center_x, it.bbox.center_y), reverse=False))


class Grp:
    def __init__(self, x_threshold: float, y_threshold: float, items=None):
        if items is None:
            items = []

        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.items = items

    @staticmethod
    def __default_grp_predicate(items, item) -> bool:
        return True

    def accept(self, item: PredictionItem) -> bool:
        if len(self.items) == 0:
            return True
        last_in_grp = self.items[-1]

        return abs(last_in_grp.bbox.center_x - item.bbox.center_x) <= self.x_threshold \
               and abs(last_in_grp.bbox.center_y - item.bbox.center_y) <= self.y_threshold

    def append(self, item: PredictionItem):
        self.items.append(item)

    def to_pred_item(self, items_selector=None) -> PredictionItem:
        items = self.items
        if items_selector is not None:
            items = items_selector(items)

        x_min = int(sorted(map(lambda item: item.bbox.x_min, items), key=lambda it: it)[0])
        x_max = int(sorted(map(lambda item: item.bbox.x_max, items), key=lambda it: it)[-1])
        y_min = int(sorted(map(lambda item: item.bbox.y_min, items), key=lambda it: it)[0])
        y_max = int(sorted(map(lambda item: item.bbox.y_max, items), key=lambda it: it)[-1])
        label = " ".join(map(lambda it: it.label, items))
        bbox = Rect((x_min, y_min, x_max, y_max))
        return PredictionItem(label=label, label_id=-1, score=1, bbox=bbox)
