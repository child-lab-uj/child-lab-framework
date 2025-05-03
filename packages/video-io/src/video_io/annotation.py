from typing import Annotated, Literal, Protocol, Self

import cv2 as opencv
import numpy
from annotated_types import Ge, Lt

type Byte = Annotated[int, Ge(0), Lt(255)]
type Color = tuple[Byte, Byte, Byte]

type RgbFrame = numpy.ndarray[tuple[int, int, Literal[3]], numpy.dtype[numpy.uint8]]

WHITE: Color = (255, 255, 255)
GREEN: Color = (0, 255, 0)
DARK_GRAY: Color = (90, 90, 90)


def draw_point_with_description(
    frame: RgbFrame,
    point: tuple[int, int],
    text: str,
    *,
    point_radius: int = 1,
    point_color: Color = GREEN,
    text_location: Literal['above', 'below'] = 'above',
    text_from_point_offset: int = 10,
    font: int = opencv.FONT_HERSHEY_DUPLEX,
    font_scale: float = 1.0,
    font_thickness: int = 1,
    font_color: Color = WHITE,
    box_color: Color = DARK_GRAY,
    box_opacity: float = 0.7,
    box_margin: int = 4,
) -> RgbFrame:
    opencv.circle(frame, point, point_radius, point_color, point_radius * 2)

    frame_height, frame_width, _ = frame.shape

    (text_width, text_height), _ = opencv.getTextSize(
        text,
        font,
        font_scale,
        font_thickness,
    )

    match text_location:
        case 'above':
            text_y_offset = text_height - 2 * box_margin - text_from_point_offset
            y_min = text_height + box_margin
            y_max = frame_height

        case 'below':
            text_y_offset = text_height + 2 * box_margin + text_from_point_offset
            y_min = 0
            y_max = frame_height - (text_height + box_margin)

    x_min = text_width // 2
    x_max = frame_width - x_min

    text_x = __clip(point[0] - text_width // 2, x_min, x_max)
    text_y = __clip(point[1] + text_y_offset, y_min, y_max)

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
    frame: RgbFrame,
    text: str,
    position: tuple[int, int],
    *,
    font: int = opencv.FONT_HERSHEY_DUPLEX,
    font_scale: float = 1.0,
    font_thickness: int = 1,
    font_color: Color = WHITE,
    box_color: Color = DARK_GRAY,
    box_opacity: float = 0.7,
    box_margin: int = 4,
) -> RgbFrame:
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

    rectangle_image = numpy.full(box_sub_image.shape, box_color, dtype=numpy.uint8)

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
    frame: RgbFrame,
    vertices: numpy.ndarray[tuple[int, Literal[2]], numpy.dtype[numpy.int32]],
    text: str,
    *,
    area_color: Color = GREEN,
    area_opacity: float = 0.5,
    font: int = opencv.FONT_HERSHEY_DUPLEX,
    font_color: Color = WHITE,
    font_scale: float = 1.0,
    font_thickness: int = 1,
    box_color: Color = DARK_GRAY,
    box_opacity: float = 0.7,
    box_margin: int = 4,
) -> RgbFrame:
    draw_filled_polygon_with_opacity(
        frame,
        vertices,
        color=area_color,
        opacity=area_opacity,
    )

    text_width = opencv.getTextSize(text, font, font_scale, font_thickness)[0][0]

    text_x: int
    text_y: int
    text_x, text_y = numpy.mean(vertices, axis=0).astype(int).tolist()

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
    frame: RgbFrame,
    vertices: numpy.ndarray[tuple[int, Literal[2]], numpy.dtype[numpy.int32]],
    *,
    color: Color = GREEN,
    opacity: float = 0.7,
) -> RgbFrame:
    solid_color = numpy.zeros_like(frame, dtype=numpy.uint8)
    solid_color[:] = numpy.array(color, dtype=numpy.uint8)

    mask = numpy.zeros_like(frame, dtype=numpy.uint8)
    opencv.fillPoly(mask, [vertices], (255, 255, 255))
    negative_mask = numpy.full_like(mask, 255) - mask

    colored_polygon = opencv.bitwise_and(solid_color, mask)
    polygon_on_frame = opencv.addWeighted(
        colored_polygon,
        opacity,
        frame,
        1 - opacity,
        0,
    )

    opencv.bitwise_or(
        opencv.bitwise_and(frame, negative_mask),
        opencv.bitwise_and(polygon_on_frame, mask),
        frame,
    )

    return frame


class Comparable(Protocol):
    def __lt__(self, _other: Self, /) -> bool: ...
    def __gt__(self, _other: Self, /) -> bool: ...


def __clip[T: Comparable](value: T, min_value: T, max_value: T) -> T:
    return max(min_value, min(value, max_value))
