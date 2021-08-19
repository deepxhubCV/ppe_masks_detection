import argparse
import os
import cv2
import sys
import time
import torch
import numpy as np

sys.path.append('./yolov5')
from utils.plots import plot_one_box
from utils.datasets import LoadImages

import norfair
from norfair import Detection, Tracker, Color

from helpers import Model
from helpers import xyxy_to_det_arr, det_arr_to_xyxy
from helpers import bboxes_iou, xyxy2xywh, iou_distance


CLS_TO_COLOR = {0: Color.green, 1: Color.red, 2: Color.yellow}


def write_image(im0, save_path, vid_path, vid_cap, vid_writer):
    if vid_path != save_path:
        vid_path = save_path
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()

        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    vid_writer.write(im0)
    return vid_path, vid_writer


def detect(source, output_path, weights, device='cuda:0', img_size=1280, conf_thres=0.5):
    t0 = time.time()

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    _, video_name = os.path.split(source)
    save_path = os.path.join(output_path, video_name)

    dataset = LoadImages(source, img_size=img_size)

    fps = dataset.cap.get(cv2.CAP_PROP_FPS)
    w = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    max_distance_between_points = 0.8  # max iou distance for matching
    tracker = Tracker(
        distance_function=iou_distance,
        distance_threshold=max_distance_between_points,
        hit_inertia_max=20,
        initialization_delay=8
    )

    model = Model(weights, img_size=img_size, device=device, conf_thres=conf_thres)

    for frame_id, (path, img, im0s, vid_cap) in enumerate(dataset):
        draw_img = im0s.copy()

        det = model.predict(img, im0s)

        masks_detections = list()
        if det is not None:
            for i, (*xyxy, conf, cls) in enumerate(det):
                xyxy = list(map(int, xyxy))
                masks_detections.append(Detection(xyxy_to_det_arr(xyxy), data=int(cls)))

        masks_tracked = tracker.update(masks_detections)

        for mask in masks_tracked:
            cls = mask.last_detection.data
            xyxy = det_arr_to_xyxy(mask.estimate)

            plot_one_box(xyxy, draw_img, label=model.names[cls], color=CLS_TO_COLOR[cls], line_width=1)

        vid_writer.write(draw_img)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='source')
    parser.add_argument('--output_path', type=str, default='output', help='output folder')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default='masks.pt', help='model weights')
    parser.add_argument('--img-size', type=int, default=1280, help='inference image size')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='inference confidence threshold')

    args = parser.parse_args()
    print(args)


    with torch.no_grad():
        detect(source=args.source, output_path=args.output_path, weights=args.weights, device=args.device,
               img_size=args.img_size, conf_thres=args.conf_thres)
