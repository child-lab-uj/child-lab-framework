from typing import Literal

import cv2 as opencv
import numpy as np

from ...typing.video import Frame


def draw_point_with_description(
    frame: Frame,
    point: tuple[int, int],
    text: str,
    *,
    point_radius: int = 3,
    point_color: tuple[int, int, int] = (0, 255, 0),
    text_location: Literal['above', 'below'] = 'below',
    text_from_point_offset: int = 5,
    font: int = opencv.FONT_HERSHEY_DUPLEX,
    font_scale: float = 1.0,
    font_thickness: int = 1,
    font_color: tuple[int, int, int] = (255, 255, 255),
    box_color: tuple[int, int, int] = (90, 90, 90),
    box_opacity: float = 0.7,
    box_margin: int = 4,
) -> Frame:
    opencv.circle(frame, point, point_radius, point_color, point_radius * 2)

    (text_width, text_height), _ = opencv.getTextSize(
        text,
        font,
        font_scale,
        font_thickness,
    )

    match text_location:
        case 'above':
            text_y_offset = text_height - 2 * box_margin - text_from_point_offset

        case 'below':
            text_y_offset = text_height + 2 * box_margin + text_from_point_offset

    text_x = point[0] - text_width // 2
    text_y = point[1] + text_y_offset

    draw_text_within_box(
        frame,
        text,
        (text_x, text_y),
        font=font,
        font_scale=font_scale,
        font_thickness=font_thickness,
        font_color=font_color,
        box_color=box_color,
        box_opacity=box_opacity,
        box_margin=box_margin,
    )

    return frame


def draw_text_within_box(
    frame: Frame,
    text: str,
    position: tuple[int, int],
    *,
    font: int = opencv.FONT_HERSHEY_DUPLEX,
    font_scale: float = 1.0,
    font_thickness: int = 1,
    font_color: tuple[int, int, int] = (255, 255, 255),
    box_color: tuple[int, int, int] = (90, 90, 90),
    box_opacity: float = 0.7,
    box_margin: int = 4,
) -> Frame:
    (text_width, text_height), _ = opencv.getTextSize(
        text,
        font,
        font_scale,
        font_thickness,
    )

    box_top_left = (
        max(text_height, position[0] - box_margin),
        max(0, position[1] - box_margin - text_height),
    )

    box_bottom_right = (
        box_top_left[0] + text_width + 2 * box_margin,
        box_top_left[1] + text_height + 2 * box_margin,
    )

    frame_height, frame_width, _ = frame.shape
    match box_bottom_right[0] >= frame_width, box_bottom_right[1] >= frame_height:
        case True, True:
            box_bottom_right = (frame_width - 1, frame_height - 1)
            box_top_left = (
                box_bottom_right[0] - text_width - 2 * box_margin,
                box_bottom_right[1] - text_height - 2 * box_margin,
            )

        case True, False:
            box_bottom_right = (frame_width - 1, box_bottom_right[1])
            box_top_left = (
                box_bottom_right[0] - text_width - 2 * box_margin,
                box_top_left[1],
            )

        case False, True:
            box_bottom_right = (box_bottom_right[0], frame_height - 1)
            box_top_left = (
                box_top_left[0],
                box_bottom_right[1] - text_height - 2 * box_margin,
            )

    box_sub_image = frame[
        box_top_left[1] : box_bottom_right[1],
        box_top_left[0] : box_bottom_right[0],
    ]

    rectangle_image = np.full(box_sub_image.shape, box_color, dtype=np.uint8)

    blended_image = opencv.addWeighted(
        box_sub_image,
        1 - box_opacity,
        rectangle_image,
        box_opacity,
        gamma=0.0,
    )

    frame[
        box_top_left[1] : box_bottom_right[1],
        box_top_left[0] : box_bottom_right[0],
    ] = blended_image

    opencv.putText(
        frame,
        text,
        position,
        font,
        font_scale,
        font_color,
        font_thickness,
        lineType=opencv.LINE_AA,
    )

    return frame


# TODO: Allow customising the text position
def draw_polygon_with_description(
    frame: Frame,
    vertices: np.ndarray[tuple[int, Literal[2]], np.dtype[np.int32]],
    text: str,
    *,
    area_color: tuple[int, int, int] = (0, 255, 0),
    area_opacity: float = 0.5,
    font: int = opencv.FONT_HERSHEY_DUPLEX,
    font_color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 1.0,
    font_thickness: int = 1,
    box_color: tuple[int, int, int] = (0, 255, 0),
    box_opacity: float = 0.7,
    box_margin: int = 4,
) -> Frame:
    draw_filled_polygon_with_opacity(
        frame,
        vertices,
        color=area_color,
        opacity=area_opacity,
    )

    text_width = opencv.getTextSize(text, font, font_scale, font_thickness)[0][0]

    text_x: int
    text_y: int
    text_x, text_y = np.mean(vertices, axis=0).astype(int).tolist()

    text_x -= text_width // 2

    draw_text_within_box(
        frame,
        text,
        (text_x, text_y),
        font=font,
        font_scale=font_scale,
        font_thickness=font_thickness,
        font_color=font_color,
        box_color=box_color,
        box_opacity=box_opacity,
        box_margin=box_margin,
    )

    return frame


def draw_filled_polygon_with_opacity(
    frame: Frame,
    vertices: np.ndarray[tuple[int, Literal[2]], np.dtype[np.int32]],
    *,
    color: tuple[int, int, int],
    opacity: float,
) -> Frame:
    solid_color = np.zeros_like(frame, dtype=np.uint8)
    solid_color[:] = np.array(color, dtype=np.uint8)

    mask = np.zeros_like(frame, dtype=np.uint8)
    opencv.fillPoly(mask, [vertices], (255, 255, 255))
    negative_mask = np.full_like(mask, 255) - mask

    colored_polygon = opencv.bitwise_and(solid_color, mask)
    polygon_on_frame = opencv.addWeighted(colored_polygon, opacity, frame, 1 - opacity, 0)

    opencv.bitwise_or(
        opencv.bitwise_and(frame, negative_mask),
        opencv.bitwise_and(polygon_on_frame, mask),
        frame,
    )

    return frame
