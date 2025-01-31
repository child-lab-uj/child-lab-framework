# ChildLab Framework
#### A System for Holistic Tracking of Parent-Child Pose Dynamics in Multi-Camera Environments


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

Currently the main method of using the framework is a CLI:

```bash
python -m child_lab_framework -help calibrate

python -m child_lab_framework -help estimate-transformations

python -m child_lab_framework -help process
```
