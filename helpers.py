import sys
import torch
import numpy as np

sys.path.append('./yolov5')
from models.experimental import attempt_load
from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size, non_max_suppression, scale_coords


class Model:
    """ Class to interact with YOLOv5 model """
    def __init__(self, weights, img_size=640, device='', conf_thres=0.5, iou_thres=0.5, classes=None,
                 agnostic_nms=False, augment=False):
        self._weights = weights
        self._img_size = img_size
        self._device = select_device(device)
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._classes = classes
        self._agnostic_nms = agnostic_nms
        self._augment = augment

        self._half = self._device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self._weights, map_location=self._device)  # load FP32 model
        self._img_size = check_img_size(self._img_size, s=self.model.stride.max())  # check img_size
        if self._half:
            self.model.half()  # to FP16

        self._names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self._colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self._names))]

    @property
    def names(self):
        """ Get list of names of classes that are being detected """
        return self._names

    @property
    def colors(self):
        """ Get list of colors for detected classes """
        return self._colors

    def predict(self, img, im0s):
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = self.model(img, augment=self._augment)[0]

        # Apply NMS
        # we are performing inference only on 1 frame --> return det for first image only
        det = non_max_suppression(pred, self._conf_thres, self._iou_thres, classes=self._classes,
                                  agnostic=self._agnostic_nms)[0]
        t2 = time_sync()
        print('Inference time: {:.3f}s'.format(t2 - t1))

        if det is not None and len(det):
            # Rescale boxes from img_size to im0s size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            return det


def bboxes_iou(xyxy1, xyxy2) -> float:
    """
    Calculate IoU for two bounding boxes
    :param xyxy1: array-like, contains (x1, y1, x2, y2)
    :param xyxy2: array-like, contains (x1, y1, x2, y2)
    :return: float, IoU(xyxy1, xyxy2)
    """
    x1_d, y1_d, x2_d, y2_d = xyxy1
    x1_e, y1_e, x2_e, y2_e = xyxy2

    # determine the coordinates of the intersection rectangle
    x_left = max(x1_d, x1_e)
    y_top = max(y1_d, y1_e)
    x_right = min(x2_d, x2_e)
    y_bottom = min(y2_d, y2_e)

    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    bb1_area = (x2_d - x1_d + 1) * (y2_d - y1_d + 1)
    bb2_area = (x2_e - x1_e + 1) * (y2_e - y1_e + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert 0 <= iou <= 1, f'expected value in range [0, 1], got {iou}'  # double-check ourselves

    return iou


def det_arr_to_xyxy(detection: np.ndarray) -> tuple:
    """
    Helper function to convert detection in format for norfair into bounding box
    :param detection: np.ndarray, array([[x1, y1], [x2, y2]])
    :return: tuple, contains (x1, y1, x2, y2)
    """
    (x1_d, y1_d), (x2_d, y2_d) = detection
    x1_d, x2_d = min(x1_d, x2_d), max(x1_d, x2_d)
    y1_d, y2_d = min(y1_d, y2_d), max(y1_d, y2_d)
    return tuple(map(int, (x1_d, y1_d, x2_d, y2_d)))


def xyxy_to_det_arr(xyxy) -> np.ndarray:
    """
    Helper function to convert detection bounding box into Detection format for norfair
    :param xyxy: array-like, contains (x1, y1, x2, y2)
    :return: np.ndarray, array([[x1, y1], [x2, y2]])
    """
    x1, y1, x2, y2 = xyxy
    return np.array([[x1, y1], [x2, y2]])


def xyxy2xywh(box):
    """
    Convert bounding box from xyxy to xywh format
    :param box: array-like, contains (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x_c = x1 + w / 2
    y_c = y1 + h / 2
    return x_c, y_c, w, h


def iou_distance(detection, tracked_object) -> float:
    """ IoU distance function for Norfair """
    return 1 - bboxes_iou(det_arr_to_xyxy(detection.points), det_arr_to_xyxy(tracked_object.estimate))