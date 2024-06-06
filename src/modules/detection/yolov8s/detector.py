import torch
import numpy as np
import cv2
from ultralytics.engine.results import Boxes
import yaml
from .utils import get_sahi_model, scale_boxes, select_device, letterbox, sahi_detect, non_max_suppression


class DetectorYolov8s:

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
        self._model = torch.load(
            cfg["model_path"], map_location=self._device)["model"].float()
        self._model.to(self._device).eval()
        self._conf_thr = cfg["conf_thr"]
        self._iou_thr = cfg["iou_thr"]
        self._input_size = cfg["input_size"]
        self.agnostic_nms = cfg["agnostic_nms"]

    def _preprocess(self, img):
        img = cv2.resize(img, (self._input_size, self._input_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self._device)
        img = img.unsqueeze(0).float()
        img /= 255.0
        return img

    def _postprocess(self, results, img_shape, img0_shape):
        results = non_max_suppression(
            results, self._conf_thr, self._iou_thr, agnostic=self.agnostic_nms)[0]
        dets = scale_boxes(img_shape, results, img0_shape)
        return dets

    def get_boxes(self, frame):
        img = self._preprocess(frame)
        results = self._model(img, augment=False)[0]
        results = self._postprocess(results, img.shape[2:4], frame.shape[:2])

        boxes = []
        scores = []
        results = results[results[:, 4] > self._conf_thr]
        boxes = results[:, :4].cpu().numpy()
        scores = results[:, 4].cpu().numpy()
        return frame, boxes, scores
