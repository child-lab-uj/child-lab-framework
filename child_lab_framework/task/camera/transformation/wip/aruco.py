import cv2


def detect_markers(
    img, aruco_dict=cv2.aruco.DICT_5X5_100, aruco_params=cv2.aruco.DetectorParameters()
):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    (corners, ids, _rejected) = detector.detectMarkers(img)
    return corners, ids
