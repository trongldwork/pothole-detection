import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def select_device(device: str):
    if device.lower() == 'cpu':
        return torch.device('cpu')
    else:
        cuda = device.isnumeric()
        if cuda:
            device = int(device)
            assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device
            return torch.device('cuda:%d' % device)
        else:
            assert False, 'Invalid device %s requested' % device


def letterbox(img, new_size):
    letterbox = LetterBox(new_size)
    img = letterbox(image=img)
    return img


def get_sahi_model(model_path, imgsz=640, conf_thr=0.25, device='cpu'):
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        image_size=imgsz,
        model_path=model_path,
        confidence_threshold=conf_thr,
        device=device,
    )
    return model


def sahi_detect(img, model, slice_w=256, slice_h=256, overlap_w_r=0.2, overlap_h_r=0.2, verbose=1):
    results = get_sliced_prediction(
        detection_model=model,
        image=img,
        slice_width=slice_w,
        slice_height=slice_h,
        overlap_width_ratio=overlap_w_r,
        overlap_height_ratio=overlap_h_r,
        verbose=verbose
    )
    return results


def scale_boxes(shape1, box, shape0):
    gain = shape0[0] / shape1[0], shape0[1] / shape1[1]
    box[:, [0, 2]] = box[:, [0, 2]] * gain[1]
    box[:, [1, 3]] = box[:, [1, 3]] * gain[0]
    return box
