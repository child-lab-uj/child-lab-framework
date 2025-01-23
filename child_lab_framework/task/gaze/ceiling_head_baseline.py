from dataclasses import dataclass

import cv2
import numpy as np

from ...core.video import Properties
from ...task import visualization
from ...task.pose import pose
from ...task.pose.keypoint import YoloKeypoint
from ...task.visualization import annotation
from ...typing.array import (
    BoolArray1,
    BoolArray3,
    ByteArray2,
    ByteArray3,
    FloatArray1,
    FloatArray2,
    FloatArray3,
    IntArray2,
)
from ...typing.video import Frame


@dataclass(slots=True)
class Result:
    actor_visibility: BoolArray1
    depth_thresholds: np.ndarray[tuple[int], np.dtype[np.float32]]
    head_masks: ByteArray3
    mask_centroids: FloatArray2
    gaze_vectors: FloatArray2

    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: visualization.Configuration,
    ) -> Frame:
        mask_opacity = 0.6
        mask_color = (255, 0, 255)

        gaze_color = (255, 0, 0)
        gaze_thickness = 3

        solid_color = np.zeros_like(frame, dtype=np.uint8)
        solid_color[:] = np.array(mask_color, dtype=np.uint8)

        mask = np.zeros_like(frame, dtype=np.uint8)
        negative_mask = np.full_like(mask, 255)

        flat_mask: ByteArray2
        for flat_mask in self.head_masks:
            flat_mask *= 255

            mask[..., 0] = flat_mask
            mask[..., 1] = flat_mask
            mask[..., 2] = flat_mask

            negative_mask[...] = 255
            negative_mask -= mask

            solid_color_region = cv2.bitwise_and(solid_color, mask)
            transparent_region_on_frame = cv2.addWeighted(
                solid_color_region,
                mask_opacity,
                frame,
                1.0 - mask_opacity,
                0,
            )

            cv2.bitwise_or(
                cv2.bitwise_and(frame, negative_mask),
                cv2.bitwise_and(transparent_region_on_frame, mask),
                frame,
            )

        for (
            is_visible,
            threshold,
            (centroid_x, centroid_y),
            (gaze_x, gaze_y),
        ) in zip(
            self.actor_visibility,
            self.depth_thresholds,
            self.mask_centroids,
            self.gaze_vectors,
        ):
            if not is_visible:
                continue

            gaze_start = (
                int(centroid_x - 400 * gaze_x),
                int(centroid_y - 400 * gaze_y),
            )

            gaze_end = (
                int(centroid_x + 400 * gaze_x),
                int(centroid_y + 400 * gaze_y),
            )

            cv2.line(
                frame,
                gaze_start,
                gaze_end,
                color=gaze_color,
                thickness=gaze_thickness,
            )

            annotation.draw_point_with_description(
                frame,
                (int(centroid_x), int(centroid_y)),
                f'threshold = {threshold:.2f}',
                font_scale=0.5,
            )

        return frame


__DENOISING_KERNEL_SMALL_SIZE = 31
__DENOISING_KERNEL_LARGE_SIZE = 31

__DENOISING_KERNEL_SMALL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (__DENOISING_KERNEL_SMALL_SIZE, __DENOISING_KERNEL_SMALL_SIZE),
)

__DENOISING_KERNEL_LARGE = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (__DENOISING_KERNEL_LARGE_SIZE, __DENOISING_KERNEL_LARGE_SIZE),
)


def estimate(
    ceiling_poses: pose.Result,
    depth: FloatArray2,
    shoulder_confidence_threshold: float,
    shoulder_depth_accuracy: float,
    adult_quantile: float = 0.4,
    child_quantile: float = 0.8,
) -> Result | None:
    shoulders: FloatArray3 = ceiling_poses.keypoints[
        :,
        [YoloKeypoint.LEFT_SHOULDER, YoloKeypoint.RIGHT_SHOULDER],
        :,
    ]

    actors = ceiling_poses.actors
    n_actors = len(actors)

    left_shoulder_confidence: FloatArray1 = shoulders[:, 0, 2]
    right_shoulder_confidence: FloatArray1 = shoulders[:, 1, 2]

    left_ge_right = left_shoulder_confidence >= right_shoulder_confidence
    left_lt_right = ~left_ge_right

    favor_shoulders = np.zeros((n_actors, 3), dtype=np.float32)
    favor_shoulders[left_ge_right, :] = shoulders[left_ge_right, 0, :]
    favor_shoulders[left_lt_right, :] = shoulders[left_lt_right, 1, :]

    actor_visibility_mask: BoolArray1 = (
        favor_shoulders[:, 2] >= shoulder_confidence_threshold
    )

    if not np.any(actor_visibility_mask):
        return None

    mean_shoulders: FloatArray2 = np.mean(shoulders[..., :2], axis=1)

    roi_radii: FloatArray1 = np.linalg.norm(
        shoulders[:, 0, :2] - mean_shoulders,
        axis=1,
    )

    for i, actor in enumerate(actors):
        if actor == pose.Actor.ADULT:
            roi_radii[i] *= 2.0
        else:
            roi_radii[i] *= 2.5

    roi_row_ranges: IntArray2 = np.clip(
        np.concatenate(
            (
                (mean_shoulders[..., 1] - roi_radii).reshape(-1, 1),
                (mean_shoulders[..., 1] + roi_radii).reshape(-1, 1),
            ),
            axis=1,
        ),
        0.0,
        1080.0 - 1.0,
    ).astype(np.int32)

    roi_column_ranges: IntArray2 = np.clip(
        np.concatenate(
            (
                (mean_shoulders[..., 0] - roi_radii).reshape(-1, 1),
                (mean_shoulders[..., 0] + roi_radii).reshape(-1, 1),
            ),
            axis=1,
        ),
        0.0,
        1920.0 - 1.0,
    ).astype(np.int32)

    roi_masks: BoolArray3 = np.zeros((n_actors, *depth.shape), dtype=np.bool_)

    for i, ((from_row, to_row), (from_column, to_column)) in enumerate(
        zip(roi_row_ranges, roi_column_ranges)
    ):
        roi_masks[i, from_row:to_row, from_column:to_column] = True

    favor_shoulder_pixel_coordinates: IntArray2 = favor_shoulders[:, :2].astype(int)
    favor_shoulder_depths: FloatArray1 = depth[
        favor_shoulder_pixel_coordinates[:, 1],
        favor_shoulder_pixel_coordinates[:, 0],
    ]

    max_head_depths: FloatArray1 = np.zeros(n_actors, dtype=np.float32)
    max_head_depths[actor_visibility_mask] = favor_shoulder_depths[actor_visibility_mask]

    # The last two singleton dimensions will be expanded during the broadcasted comparison,
    # resulting in an array of masks indicating the interesting regions for each actor.
    baseline_head_masks: BoolArray3 = depth <= max_head_depths.reshape(-1, 1, 1)
    baseline_head_masks &= roi_masks

    quantile_thresholds: FloatArray1 = np.zeros(n_actors, dtype=np.float32)

    for i in range(n_actors):
        quantile = adult_quantile if actors[0] == pose.Actor.ADULT else child_quantile

        quantile_thresholds[i] = np.quantile(
            depth[baseline_head_masks[i, ...]],
            q=quantile,
        )

    head_masks = depth <= quantile_thresholds.reshape(-1, 1, 1)
    head_masks &= roi_masks

    head_masks_smoothed: ByteArray3 = np.zeros_like(
        head_masks,
        dtype=np.uint8,
    )

    for i in range(n_actors):
        mask = head_masks[i, ...].astype(np.uint8)

        if actors[i] == pose.Actor.ADULT:
            kernel = __DENOISING_KERNEL_LARGE
            iterations = 5
        else:
            kernel = __DENOISING_KERNEL_SMALL
            iterations = 1

        head_masks_smoothed[i, ...] = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=iterations,
        )

    mask_centroids = np.zeros((n_actors, 2), dtype=np.float32)
    gaze_vectors = np.zeros((n_actors, 2), dtype=np.float32)

    for i in range(n_actors):
        if not actor_visibility_mask[i]:
            continue

        mask: ByteArray2 = head_masks_smoothed[i, ...]

        if np.sum(mask) == 0:
            continue

        mask_points: FloatArray2 = np.stack(np.where(mask != 0)[::-1]).astype(np.float32)

        mask_centroid = np.mean(mask_points, axis=1)
        mask_centroids[i, :] = mask_centroid

        mask_points_centered: FloatArray2 = mask_points - mask_centroid.reshape(-1, 1)

        covariance = np.cov(mask_points_centered)
        eigen_values, eigen_vectors = np.linalg.eig(covariance)

        gaze_vectors[i] = eigen_vectors[:, np.argmax(eigen_values)]

    return Result(
        actor_visibility_mask,
        quantile_thresholds,
        head_masks_smoothed,
        mask_centroids,
        gaze_vectors,
    )
