import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from child_lab_framework.core.calibration import Calibration
from child_lab_framework.core.video import Format, Input, Properties, Reader, Writer
from child_lab_framework.task import depth
from child_lab_framework.task.camera.detection import marker
from child_lab_framework.task.visualization import annotation, visualization
from child_lab_framework.task.visualization.visualization import Visualizer

FOV_LOW = 4000.0
FOV_HIGH = 5000.0
FOV_STEP = 100.0

fov_x_range = np.arange(FOV_LOW, FOV_HIGH, step=FOV_STEP, dtype=np.float32)
fov_y_range = np.arange(FOV_LOW, FOV_HIGH, step=FOV_STEP, dtype=np.float32)

source = Path(__file__).parent / sys.argv[1]
assert source.is_file()

destination = source.parent / source.name

calibration = Calibration.heuristic(1080, 1920)

properties = Properties(
    'wideo',
    0,
    1080,
    1920,
    50,
    calibration,
)

reader = Reader(Input('wideo', source, None), batch_size=1)
frame_og = reader.read()
assert frame_og is not None
del reader

frame_og.flags.writeable = True
frame_ulep_og = frame_og.copy()

visualizer = Visualizer(
    properties=properties,
    configuration=visualization.Configuration(),
)

detector = marker.Detector(
    model=marker.RigidModel(50.0, 0.0),
    dictionary=marker.Dictionary.ARUCO_5X5_1000,
    detector_parameters=cv2.aruco.DetectorParameters(),
)

writer = Writer(
    destination,
    properties,
    output_format=Format.MP4,
)

writer_ulep = Writer(
    destination.parent / f'{destination.stem}_ulep.mp4',
    properties,
    output_format=Format.MP4,
)

device = torch.device('cuda')
depth_estimator = depth.Estimator(device)


for fov_x in fov_x_range:
    for fov_y in fov_y_range:
        calibration.focal_length = (float(fov_x), float(fov_y))

        properties.calibration = calibration

        detector.calibration = calibration

        frame_depth = depth_estimator.predict(frame_og, properties)
        markers, markers_ulep = detector.predict(frame_og, frame_depth)  # type: ignore # help pls

        if markers is not None:
            frame = visualizer.annotate(frame_og, markers)
            frame_ulep = visualizer.annotate(frame_ulep_og, markers_ulep)
            print('[Step complete]')

        else:
            frame = frame_og.copy()
            frame_ulep = frame_ulep_og.copy()
            print('[No detection]')

        frame = annotation.draw_text_within_box(
            frame,
            f'FOV ({fov_x:.2f}, {fov_y:.2f})',
            (960, 30),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=1.0,
            font_thickness=2,
            font_color=(255, 255, 255),
            box_color=(90, 90, 90),
            box_opacity=0.7,
            box_margin=4,
        )

        frame_ulep = annotation.draw_text_within_box(
            frame_ulep,
            f'Ülep FOV ({fov_x:.2f}, {fov_y:.2f})',
            (960, 30),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=1.0,
            font_thickness=2,
            font_color=(255, 255, 255),
            box_color=(90, 90, 90),
            box_opacity=0.7,
            box_margin=4,
        )

        writer.write(frame)
        writer_ulep.write(frame_ulep)
