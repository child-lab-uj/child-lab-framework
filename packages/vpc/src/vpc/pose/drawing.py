import typing

import cv2 as opencv
import numpy
from jaxtyping import Float
from video_io.annotation import Color
from video_io.frame import ArrayRgbFrame

from .model import YOLO_SKELETON


def draw_keypoints(
    frame: ArrayRgbFrame,
    keypoints: Float[numpy.ndarray, 'n_detections 17 3'],
    bone_color: Color,
    bone_thickness: int,
    keypoint_color: Color,
    keypoint_radius: int,
    keypoint_min_confidence: float,
) -> None:
    actor_keypoints: numpy.ndarray
    for actor_keypoints in keypoints:
        for i, j in YOLO_SKELETON:
            if actor_keypoints[i, -1] < keypoint_min_confidence:
                continue

            if actor_keypoints[j, -1] < keypoint_min_confidence:
                continue

            start = typing.cast(opencv.typing.Point, actor_keypoints[i, :-1].astype(int))
            end = typing.cast(opencv.typing.Point, actor_keypoints[j, :-1].astype(int))

            opencv.line(frame, start, end, bone_color, bone_thickness)

        for keypoint in actor_keypoints:
            if keypoint[-1] < keypoint_min_confidence:
                continue

            keypoint = typing.cast(opencv.typing.Point, keypoint.astype(int))
            opencv.circle(frame, keypoint[:-1], keypoint_radius, keypoint_color, -1)


def draw_bounding_boxes(
    frame: ArrayRgbFrame,
    boxes: Float[numpy.ndarray, 'n_detections 5'],
    color: Color,
    thickness: int,
    min_confidence: float,
) -> None:
    box: numpy.ndarray
    for box in boxes.astype(int):
        x1, y1, x2, y2, confidence, *_ = box

        if confidence < min_confidence:
            continue

        opencv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
