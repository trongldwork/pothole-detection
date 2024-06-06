import torch
import numpy as np
import cv2
from ultralytics.engine.results import Boxes
import yaml
from .utils import get_sahi_model, scale_boxes, select_device, letterbox, sahi_detect, non_max_suppression


class DetectorSAHI:

    def __init__(self, model_config="yolov8s.yaml") -> None:
        try:
            with open("configs\Models/" + model_config) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model configuration file not found: {model_config}")
        self._cfg = cfg
        torch.backends.cudnn.benchmark = True
        self._device = select_device(str(cfg["gpu_id"]))
        self._conf_thr = cfg["conf_thr"]
        self._iou_thr = cfg["iou_thr"]
        self._input_size = cfg["input_size"]
        self.agnostic_nms = cfg["agnostic_nms"]
        self._model = get_sahi_model(
            model_path=cfg["model_path"], imgsz=cfg["input_size"], conf_thr=cfg["conf_thr"], device=self._device)

    def get_boxes(self, frame):
        results = sahi_detect(frame, model=self._model, slice_w=512,
                              slice_h=512, overlap_w_r=0.2, overlap_h_r=0.2, verbose=0).object_prediction_list

        boxes = []
        scores = []

        for result in results:
            if result.score.is_greater_than_threshold(self._conf_thr):
                box = result.bbox.to_xyxy()
                # print(box)
                # box = scale_boxes(
                #     (result.full_shape[0], result.full_shape[1]), box, frame.shape)
                boxes.append(box)
                scores.append(result.score.value)
        return frame, boxes, scores
