import cv2 as opencv
import numpy
from jaxtyping import Float, Int
from video_io import annotation
from video_io.frame import ArrayRgbFrame

from marker_detection.aruco.geometry import (
    IntrinsicsMatrix,
    MarkerRigidModel,
    Transformation,
)


def draw_3d_dice_models(
    frame: ArrayRgbFrame,
    marker_transformations: list[Transformation],
    intrinsics: IntrinsicsMatrix,
    marker_rigid_model: MarkerRigidModel,
) -> None:
    marker_size = marker_rigid_model.square_size
    marker_border = marker_rigid_model.border

    min_x = min_y = -marker_size / 2 - marker_border
    max_x = max_y = marker_size / 2 + marker_border
    min_z = -marker_size - 2 * marker_border
    max_z = 0.0

    # A rigid model of the AruDie seen in front,
    # with the detected code on the 0-1-2-3 wall:
    #
    #   | Y
    #   |
    #   | 7 ----- 6
    #   |/|      /|
    #   3 ----- 2 |
    #   | 4 ----| 5
    #   |/      |/    X
    #   0 ----- 1 -----
    #  /
    # / Z
    #
    # Upper layer:
    # 3 - (min_x, max_y, max_z)  2 - (max_x, max_y, max_z)
    # 0 - (min_x, min_y, max_z)  1 - (max_x, min_y, max_z)
    #
    # Lower layer:
    # 7 - (min_x, max_y, min_z)  6 - (max_x, max_y, min_z)
    # 4 - (min_x, min_y, min_z)  5 - (max_x, min_y, min_z)

    cube_points3d: Float[numpy.ndarray, '3 8'] = numpy.array(
        [
            # Upper:
            [min_x, min_y, max_z],  # 0
            [max_x, min_y, max_z],  # 1
            [max_x, max_y, max_z],  # 2
            [min_x, max_y, max_z],  # 3
            # Lower:
            [min_x, min_y, min_z],  # 4
            [max_x, min_y, min_z],  # 5
            [max_x, max_y, min_z],  # 6
            [min_x, max_y, min_z],  # 7
        ]
    ).T

    cube_edges = (
        # Upper layer:
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        # Lower layer:
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        # Inter-layer:
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    for transformation in marker_transformations:
        transformed_cube = (
            transformation.rotation @ cube_points3d
            + transformation.translation.reshape(-1, 1)
        )

        zs = transformed_cube[-1, :].copy()
        transformed_cube /= zs

        projected_cube = intrinsics @ transformed_cube
        projected_cube_corners: list[tuple[int, int]] = [
            tuple(point) for point in projected_cube[:-1, :].astype(numpy.int32).T
        ]

        for i, j in cube_edges:
            opencv.line(
                frame,
                projected_cube_corners[i],
                projected_cube_corners[j],
                (255, 0, 0),
                2,
            )

        for projected_cube_corner, z in zip(projected_cube_corners, zs):
            annotation.draw_point_with_description(
                frame,
                projected_cube_corner,
                f'{z:.2f}',
                point_radius=2,
                point_color=(255, 0, 0),
                font_scale=0.25,
            )

        break


def draw_marker_masks(
    frame: ArrayRgbFrame,
    marker_corners: Int[numpy.ndarray, 'n 4 2'],
    color_with_alpha: tuple[float, float, float, float],
) -> None:
    r, g, b, alpha = color_with_alpha
    color = (int(r), int(g), int(b))

    marker_corners_flat: Int[numpy.ndarray, '4 2']

    for marker_corners_flat in marker_corners:
        marker_corners = marker_corners_flat.reshape(-1, 2).astype(numpy.int32)

        annotation.draw_filled_polygon_with_opacity(
            frame,
            marker_corners,
            color=color,
            opacity=alpha,
        )

        corner_pixels: list[tuple[int, int]] = list(map(tuple, marker_corners))
        upper_left, upper_right, lower_right, lower_left = corner_pixels

        annotation.draw_point_with_description(
            frame,
            upper_left,
            'upper_left',
            point_radius=1,
            font_scale=0.4,
            text_location='above',
        )
        annotation.draw_point_with_description(
            frame,
            upper_right,
            'upper_right',
            point_radius=1,
            font_scale=0.4,
            text_location='above',
        )
        annotation.draw_point_with_description(
            frame,
            lower_right,
            'lower_right',
            point_radius=1,
            font_scale=0.4,
        )
        annotation.draw_point_with_description(
            frame,
            lower_left,
            'lower_left',
            point_radius=1,
            font_scale=0.4,
        )


def draw_marker_frame_axes(
    frame: ArrayRgbFrame,
    marker_transformations: list[Transformation],
    intrinsics: IntrinsicsMatrix,
    axis_length: float,
    axis_thickness: int,
    draw_angles: bool,
) -> None:
    x_color = (255, 0, 0)
    y_color = (0, 255, 0)
    z_color = (0, 0, 255)

    basis_points = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=numpy.float32,
    ).T

    for transformation in marker_transformations:
        rotation = transformation.rotation
        translation = transformation.translation.reshape(-1, 1)

        transformed_basis = rotation @ basis_points + translation
        transformed_basis /= transformed_basis[2]
        projected_basis = intrinsics @ transformed_basis

        projected_basis_pixel_coordinates: list[tuple[int, int]] = (
            projected_basis.T[:, :2].astype(int).tolist()
        )
        o, x, y, z = projected_basis_pixel_coordinates

        opencv.line(frame, o, x, x_color, axis_thickness)
        opencv.line(frame, o, y, y_color, axis_thickness)
        opencv.line(frame, o, z, z_color, axis_thickness)

        if draw_angles:
            angles: list[float] = (
                180.0 / numpy.pi * transformation.euler_angles()
            ).tolist()

            x_angle, y_angle, z_angle = angles

            annotation.draw_point_with_description(
                frame,
                x,
                f'x: {x_angle:.2f}',
                font_scale=0.3,
                point_radius=1,
                point_color=x_color,
            )
            annotation.draw_point_with_description(
                frame,
                y,
                f'y: {y_angle:.2f}',
                font_scale=0.3,
                point_radius=1,
                point_color=y_color,
            )
            annotation.draw_point_with_description(
                frame,
                z,
                f'z: {z_angle:.2f}',
                font_scale=0.3,
                point_radius=1,
                point_color=z_color,
            )

        break


def draw_marker_ids(
    frame: ArrayRgbFrame,
    marker_ids: Int[numpy.ndarray, ' n'],
    marker_corners: Int[numpy.ndarray, 'n 4 2'],
) -> None:
    for id, marker_corners in zip(marker_ids, marker_corners):
        marker_corners = marker_corners.reshape(-1, 2)
        center: tuple[int, int] = numpy.mean(marker_corners, axis=0).astype(int).tolist()

        annotation.draw_text_within_box(
            frame,
            f'marker {id}',
            center,
            font_scale=0.3,
        )
