from enum import IntEnum


class YoloKeypoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


YOLO_SKELETON = (
    (YoloKeypoint.LEFT_SHOULDER, YoloKeypoint.RIGHT_SHOULDER),
    (YoloKeypoint.LEFT_SHOULDER, YoloKeypoint.LEFT_ELBOW),
    (YoloKeypoint.RIGHT_SHOULDER, YoloKeypoint.RIGHT_ELBOW),
    (YoloKeypoint.LEFT_ELBOW, YoloKeypoint.LEFT_WRIST),
    (YoloKeypoint.RIGHT_ELBOW, YoloKeypoint.RIGHT_WRIST),
    (YoloKeypoint.LEFT_SHOULDER, YoloKeypoint.LEFT_HIP),
    (YoloKeypoint.RIGHT_SHOULDER, YoloKeypoint.RIGHT_HIP),
    (YoloKeypoint.LEFT_HIP, YoloKeypoint.RIGHT_HIP),
    (YoloKeypoint.LEFT_HIP, YoloKeypoint.LEFT_KNEE),
    (YoloKeypoint.RIGHT_HIP, YoloKeypoint.RIGHT_KNEE),
    (YoloKeypoint.LEFT_KNEE, YoloKeypoint.LEFT_ANKLE),
    (YoloKeypoint.RIGHT_KNEE, YoloKeypoint.RIGHT_ANKLE),
)
