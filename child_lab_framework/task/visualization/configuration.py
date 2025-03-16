from dataclasses import dataclass

type Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class Configuration:
    face_draw_bounding_boxes: bool = True
    face_bounding_box_color: Color = (255.0, 0.0, 0.0, 1.0)
    face_bounding_box_thickness: int = 4
    face_draw_confidence: bool = False
    face_confidence_threshold: float = 0.5
    face_confidence_color: Color = (255.0, 255.0, 255.0, 1.0)
    face_confidence_size: int = 10

    gaze_draw_lines: bool = True
    gaze_line_color: Color = (255.0, 255.0, 0.0, 1.0)
    gaze_baseline_line_color: Color = (0.0, 255.0, 255.0, 1.0)
    gaze_line_thickness: int = 5
    gaze_line_length: int = 250

    pose_draw_skeletons: bool = True
    pose_draw_skeletons_confidence: bool = True
    pose_bone_color: Color = (0.0, 0.0, 255.0, 1.0)
    pose_bone_thickness: int = 2
    pose_keypoint_color: Color = (0.0, 255.0, 0.0, 1.0)
    pose_keypoint_radius: int = 5
    pose_keypoint_confidence_threshold: float = 0.5
    pose_keypoint_confidence_color: Color = (255.0, 255.0, 255.0, 1.0)
    pose_keypoint_confidence_size: int = 10
    pose_draw_bounding_boxes: bool = True
    pose_draw_bounding_boxes_confidence: bool = True
    pose_bounding_box_color: Color = (0.0, 255.0, 0.0, 1.0)
    pose_bounding_box_thickness: int = 4
    pose_bounding_box_confidence_threshold: float = 0.5
    pose_bounding_box_confidence_color: Color = (255.0, 255.0, 255.0, 1.0)
    pose_bounding_box_confidence_size: int = 10

    social_proximity_draw_lines: bool = True
    social_proximity_line_color: Color = (0.0, 0.0, 255.0, 1.0)
    social_proximity_line_thickness: int = 3
    social_proximity_draw_distance_value: bool = True
    social_proximity_distance_value_color: Color = (255.0, 255.0, 0.0, 1.0)
    social_proximity_distance_value_size: int = 10

    chessboard_draw_corners: bool = True
    chessboard_corner_color: Color = (255.0, 0.0, 0.0, 1.0)
    chessboard_corner_radius: int = 3

    marker_draw_ids: bool = True
    marker_draw_masks: bool = True
    marker_mask_color: Color = (0.0, 255.0, 0.0, 1.0)
    marker_draw_axes: bool = True
    marker_axis_length: int = 100
    marker_axis_thickness: int = 2
    marker_draw_angles: bool = True
