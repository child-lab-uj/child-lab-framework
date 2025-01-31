# ChildLab Framework
#### A System for Holistic Tracking of Parent-Child Pose Dynamics in Multi-Camera Environments

> **Disclaimer**  
> The framework is in a very early development status.
> We plan stabilization of public APIs after the first release.

## Introduction & Motivation

###### This project is a collaborative effort of researchers at the [Jagiellonian University](https://en.uj.edu.pl) and the [AGH University](https://www.agh.edu.pl/en).

[ChildLab](http://www.labdziecka.psychologia.uj.edu.pl/o-laboratorium.html) is an organization at the [JU Institute of Psychology](https://psychologia.uj.edu.pl/en_GB/start) dedicated to researching children's psychological development. This framework serves as a tool assisting the examination of parent-child interactions. It automates laborious manual data labeling thus allowing the researchers to focus on the proper analysis.

![ChildLab Logo](./readme/childlab_logo.png)

## Overview

The main features of the framework are:

###### Modular Structure

ChildLab Framework is designed in a component-based manner. You can freely match them and create complex pipelines, called procedures.

[Examples](https://github.com/child-lab-uj/child-lab-framework/tree/main/child_lab_framework/_procedure)

###### 3D & Multi-Camera Support

Observing scenes from multiple angles and using information between the frames of reference was always crucial for the project. ChildLab Framework provides methods of camera calibration, 3D transformation estimation and result visualization.

###### Pose Estimation

One of the main components is a [YOLO](https://docs.ultralytics.com/)-based pose detection. It can serve as a basis for gesture detection, classification, movement tracking and many other common use cases.

###### Gaze Analysis

The gaze analysis component is based on [mini-face](https://github.com/child-lab-uj/mini-face) our lightweight wrapper over the [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) library. It allows for easy and comprehensive estimation with a simple Python interface.

###### Joint Attention Detection

ChildLab Framework provides various novel methods of detecting joint attention - a shared focus of one or more individuals on some other object. It aims to fill the current research gaps in 3D attention tracking.

## Usage

For the basic usage ChildLab Framework provides a CLI:

```bash
python3 -m child_lab_framework calibrate --help

python3 -m child_lab_framework estimate-transformations --help

python3 -m child_lab_framework process --help
```

The framework's components are exposed as a public API.
Below you can find an example usage, for a fuller picture please refer to the [demo](https://github.com/child-lab-uj/child-lab-framework/tree/main/child_lab_framework/_procedure/demo_sequential.py).

```python
from pathlib import Path

from child_lab_framework.core import video
from child_lab_framework.postprocessing import imputed_with_closest_known_reference
from child_lab_framework.task import pose, face, gaze, visualization


# Create video I/O components
reader = video.Reader(
    video.Input(
        'my_video',
        Path('my_video.mp4'),
        calibration=None
    ),
    batch_size=10,
)
video_properties = reader.properties

writer = video.Writer(
    Path('output.mp4'),
    video_properties,
    output_format=video.Format.MP4,
)

# Create a visualization component
visualizer = visualization.Visualizer(
    properties=video_properties,
    configuration=visualization.VisualizationConfiguration(),
)

# Perform computation on a GPU
device = torch.device('cuda')

# Prepare estimators
pose_estimator = pose.Estimator(
    device,
    input=video_properties,
    max_detections=2,
    threshold=0.5,
)

face_estimator = face.Estimator(
    device,
    input=video_properties,
    confidence_threshold=0.5,
    suppression_threshold=0.1,
)

gaze_estimator = gaze.Estimator(input=video_properties)

# Read frames
frames = reader.read_batch()
assert frames is not None

# Predict and impute poses
poses = imputed_with_closest_known_reference(
    pose_estimator.predict_batch(frames)
)

# Predict and impute faces matched with poses
faces = (
    imputed_with_closest_known_reference(
        face_estimator.predict_batch(frames, poses)
    )
    if poses is not None
    else None
)

# Estimate and impute gaze based on faces
gazes = (
    imputed_with_closest_known_reference(
        gaze_estimator.predict_batch(
            frames,
            faces,
        )
    )
    if faces is not None
    else None
)

# Visualize the results
annotated_frames = visualizer.annotate_batch(frames, poses, gazes)

# Save the results
writer.write_batch(annotated_frames)
```

## Project Roadmap

- [x] Basic framework structure
- [x] Asynchronous processing
- [x] Camera calibration
- [x] Depth estimation
- [x] 3D transformation estimation
- [x] Pose estimation
- [x] Face estimation
- [x] Gaze analysis
- [ ] Joint attention detection
- [ ] 3D visualization
- [ ] Emotion recognition
- [ ] General web GUI
- [ ] PyPI deployment
