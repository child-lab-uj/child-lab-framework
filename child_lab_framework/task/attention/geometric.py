from dataclasses import dataclass
from itertools import count
from typing import ClassVar, Literal

import cv2
import numpy as np
from icecream import ic

from ...core.video import Properties
from ...typing.video import Frame
from .. import gaze
from ..gaze import ceiling_projection
from ..visualization import Configuration, annotation


@dataclass(frozen=True, slots=True)
class Result:
    @dataclass(frozen=True, slots=True)
    class MutualGaze:
        LABEL: ClassVar[str] = 'mutual_gaze'

        convergence_angle: float
        ray_distance: float
        head_angular_velocities: np.ndarray[tuple[Literal[2]], np.dtype[np.float32]]
        diadic_divergence: float

    @dataclass(frozen=True, slots=True)
    class JointAttention:
        LABEL: ClassVar[str] = 'joint_attention'

        attention_target: np.ndarray[tuple[Literal[2]], np.dtype[np.float32]]
        target_scene_distance: float
        target_linear_velocity: float
        intersection_angle: float
        triadic_divergence: float

    timestamp: int
    event: MutualGaze | JointAttention

    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: Configuration,
    ) -> Frame:
        match self.event:
            case self.MutualGaze(angle, distance, _head_velocities, divergence):
                annotation.draw_text_within_box(
                    frame,
                    f'Mutual gaze ({self.timestamp / 50 :.2f} s): angle = {angle * 180.0 / np.pi:.2f} deg, {distance = :.2f}, {divergence = :.2f}',
                    (25, 1000),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=1.0,
                    font_thickness=1,
                    font_color=(255, 255, 255),
                    box_color=(90, 90, 90),
                    box_opacity=0.7,
                    box_margin=4,
                )

            case self.JointAttention(
                target,
                scene_distance,
                velocity,
                angle,
                divergence,
            ):
                annotation.draw_text_within_box(
                    frame,
                    f'Joint attention ({self.timestamp / 50 :.2f} s): {velocity = :.2f}, {scene_distance = :.2f}, {divergence = :.2f}',
                    (75, 1000),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=1.0,
                    font_thickness=1,
                    font_color=(255, 255, 255),
                    box_color=(90, 90, 90),
                    box_opacity=0.7,
                    box_margin=4,
                )
                annotation.draw_point_with_description(
                    frame,
                    target.astype(int).tolist(),
                    'Target',
                )

        return frame


class Estimator:
    max_parallel_angle: float
    max_parallel_distance: float
    max_target_linear_velocity: float
    max_head_angular_velocity: float
    max_scene_distance: float
    max_diadic_divergence: float
    max_triadic_divergence: float

    __time: 'count[int]'

    def __init__(
        self,
        max_parallel_angle: float,
        max_parallel_distance: float,
        max_target_linear_velocity: float,
        max_head_angular_velocity: float,
        max_scene_distance: float,
        max_diadic_divergence: float,
        max_triadic_divergence: float,
    ) -> None:
        self.max_parallel_angle = max_parallel_angle
        self.max_parallel_distance = max_parallel_distance
        self.max_target_linear_velocity = max_target_linear_velocity
        self.max_head_angular_velocity = max_head_angular_velocity
        self.max_scene_distance = max_scene_distance
        self.max_diadic_divergence = max_diadic_divergence
        self.max_triadic_divergence = max_triadic_divergence

        self.__time = count()

    # TODO:
    #   * [x] Add frame timestamps
    #   * [x] Intersect only one-sided *rays* of the gaze
    #   * [x] Detect parallel rays (cross product approximately 0)
    #     * [x] Detect mutual gaze based on the distance between approximately parallel rays
    #   * [x] Constrain the area in space in which rays can intersect
    #   * [x] Constrain the maximal velocity of the intersection point to dectect "stable" locations
    #   * [x] Add 3D information from side-view gaze directions:
    #     * [x] Constrain the maximal difference of the angles between the gaze and the horizontal direction
    #           to avoid divergence between pairs of rays
    #   * [ ] Compute and constrain angular velocities of heads, based on gaze directions
    #
    def predict_batch(
        self,
        ceiling_gazes: list[ceiling_projection.Result],
        left_gazes: list[gaze.Result3d],
        right_gazes: list[gaze.Result3d],
    ) -> list[Result | None]:
        n_observations = len(ceiling_gazes)

        gaze_yaw_angles = np.zeros((n_observations, 2), dtype=np.float32)
        intersections = np.zeros((n_observations, 2), dtype=np.float32)

        results: list[Result | None] = [None for _ in range(n_observations)]

        max_parallel_angle = self.max_parallel_angle
        max_parallel_distance = self.max_parallel_distance
        max_target_linear_velocity = self.max_target_linear_velocity
        max_head_angular_velocity = self.max_head_angular_velocity
        max_scene_distance = self.max_scene_distance
        max_diadic_divergence = self.max_diadic_divergence
        max_triadic_divergence = self.max_triadic_divergence

        for i, (ceiling_gaze, left_gaze, right_gaze) in enumerate(
            zip(
                ceiling_gazes,
                left_gazes,
                right_gazes,
            )
        ):
            timestamp = next(self.__time)

            if len(ceiling_gaze.centers) < 2:
                continue

            scene_center = ceiling_gaze.centers.mean(axis=0)
            intersection = ray_intersection_point(ceiling_gaze)
            angle = ray_intersection_angle(ceiling_gaze)
            distance = ray_distance(ceiling_gaze)

            print()
            print()

            ic(scene_center)
            ic(intersection)
            ic(angle)
            ic(distance)

            left_yaw = 0.0
            right_yaw = 0.0

            intersections[i] = intersection if intersection is not None else scene_center
            gaze_yaw_angles[i, 0] = left_yaw
            gaze_yaw_angles[i, 1] = right_yaw

            head_angular_velocities = np.abs(gaze_yaw_angles[i] - gaze_yaw_angles[i - 1])

            if np.max(head_angular_velocities) > max_head_angular_velocity:
                continue

            # +-----------------------+
            # | Check for mutual gaze |
            # +-----------------------+

            divergence = ray_diadic_divergence(left_gaze, right_gaze)

            ic(divergence)

            if (
                angle < max_parallel_angle
                and distance < max_parallel_distance
                and divergence < max_diadic_divergence
            ):
                event = Result.MutualGaze(
                    angle,
                    distance,
                    head_angular_velocities,
                    divergence,
                )
                result = Result(timestamp, event)
                ic(result)
                results[i] = result

            # +---------------------------+
            # | Check for joint attention |
            # +---------------------------+

            if intersection is None:
                continue

            scene_distance = float(np.linalg.norm(scene_center - intersection))
            ic(scene_distance)

            if scene_distance > max_scene_distance:
                continue

            intersection_velocity = float(
                np.linalg.norm(intersections[i - 1, :] - intersection)
            )

            if intersection_velocity > max_target_linear_velocity:
                continue

            divergence = ray_triadic_divergence(
                intersection,
                ceiling_gaze,
                left_gaze,
                right_gaze,
            )
            ic(divergence)

            if divergence > max_triadic_divergence:
                continue

            event = Result.JointAttention(
                intersection,
                scene_distance,
                intersection_velocity,
                angle,
                divergence,
            )
            result = Result(timestamp, event)
            ic(result)
            results[i] = result

        return results


__DETERMINANT_TOLERANCE = 1e-6


def ray_intersection_point(
    gazes: ceiling_projection.Result,
) -> np.ndarray[tuple[Literal[2]], np.dtype[np.float32]] | None:
    centers = gazes.centers
    directions = gazes.directions

    coefficients = directions.copy().transpose()
    coefficients[:, 1] *= -1.0

    main_determinant = np.linalg.det(coefficients)

    if abs(main_determinant) < __DETERMINANT_TOLERANCE:
        return None

    free_terms = centers[1, :] - centers[0, :]

    u_coefficients = coefficients.copy()
    u_coefficients[:, 0] = free_terms
    u_determinant = np.linalg.det(u_coefficients)
    u = u_determinant / main_determinant

    # v_coefficients = coefficients.copy()
    # v_coefficients[:, 1] = free_terms
    # v_determinant = np.linalg.det(v_coefficients)
    # v = v_determinant / main_determinant

    # if u * v < 0.0:
    #     return None

    return centers[0] + u * directions[0]


def ray_intersection_angle(
    gazes: ceiling_projection.Result,
) -> float:
    directions = gazes.directions
    direction1 = directions[0, :]
    direction2 = directions[1, :]

    cross_product = np.linalg.norm(np.cross(direction1, direction2))
    norm1 = np.linalg.norm(direction1)
    norm2 = np.linalg.norm(direction2)

    return float(np.arcsin(cross_product / (norm1 * norm2)))


def ray_distance(gazes: ceiling_projection.Result) -> float:
    centers = gazes.centers
    directions = gazes.directions

    a, b = directions[0]
    c, d = directions[1]

    start1_x, start1_y = centers[0]
    start2_x, start2_y = centers[1]

    tangent1 = b / a
    tangent2 = d / c

    numerator = np.abs(tangent1 * start1_x - tangent2 * start2_x - start1_y + start2_y)
    denominator = np.sqrt(tangent1**2 + 1)

    return float(numerator / denominator)


def ray_diadic_divergence(
    left_gaze: gaze.Result3d,
    right_gaze: gaze.Result3d,
) -> float:
    offset1 = left_gaze.vertical_offset_angles[0]
    offset2 = right_gaze.vertical_offset_angles[0]
    return float(np.abs(offset1 + offset2 - np.pi / 2.0))


def ray_triadic_divergence(
    intersection: np.ndarray[tuple[Literal[2]], np.dtype[np.float32]],
    ceiling_gaze: ceiling_projection.Result,
    left_gaze: gaze.Result3d,
    right_gaze: gaze.Result3d,
) -> float:
    centers = ceiling_gaze.centers
    horizontal_projection_lengths = np.linalg.norm(centers - intersection, axis=1)

    offset1 = left_gaze.vertical_offset_angles[0]
    offset2 = right_gaze.vertical_offset_angles[0]

    match ceiling_gaze.center_depths:
        case None:
            eye_height_difference = 100.0

        case heights:
            eye_height_difference = np.abs(heights[0] - heights[1])

    return float(
        np.abs(
            np.abs(
                horizontal_projection_lengths[0] * np.tan(offset1)
                - horizontal_projection_lengths[1] * np.tan(offset2)
            )
            - eye_height_difference
        )
    )
