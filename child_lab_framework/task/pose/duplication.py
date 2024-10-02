from operator import itemgetter

import torch
from ultralytics.engine import results as yolo

from ...core.geometry import area


def jaccard_similarity(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    lower_left_x = torch.max(box1[0], box2[0])
    lower_left_y = torch.max(box1[1], box2[1])
    upper_right_x = torch.min(box1[2], box2[2])
    upper_right_y = torch.min(box1[3], box2[3])

    intersection = torch.stack((lower_left_x, lower_left_y, upper_right_x, upper_right_y))

    area1 = area(box1)
    area2 = area(box2)
    intersection_area = area(intersection)

    return intersection_area / (area1 + area2 - intersection_area)


type Similarities = list[tuple[tuple[int, int], float]]


def indices_to_delete(
    boxes: list[torch.Tensor], similarities: Similarities, to_delete: int
) -> set[int]:
    candidates: set[int] = set()

    for _, ((i, j), _) in zip(range(to_delete), similarities):
        match (i in candidates, j in candidates):
            case False, False:
                candidates.add(i if boxes[i][4].item() < boxes[j][4].item() else j)

            case True, False:
                candidates.add(j)

            case False, True:
                candidates.add(i)

            case True, True:
                continue

    return candidates


def similarities(boxes: list[torch.Tensor]) -> Similarities:
    return sorted(
        [
            ((i, j), jaccard_similarity(box_i, box_j).item())
            for i, box_i in enumerate(boxes)
            for j, box_j in enumerate(boxes)
        ],
        key=itemgetter(1),
    )


def deduplicated(
    boxes: yolo.Boxes, keypoints: yolo.Keypoints, max_detections: int
) -> tuple[yolo.Boxes, yolo.Keypoints]:
    n_detections = len(boxes)
    max_detections = max_detections

    if n_detections <= max_detections:
        return boxes, keypoints

    boxes_separated = list(boxes.data)

    to_delete = indices_to_delete(
        boxes_separated, similarities(boxes_separated), n_detections - max_detections
    )

    remaining = list(set(range(n_detections)) - to_delete)

    boxes = boxes[remaining]
    keypoints = keypoints[remaining]

    return boxes, keypoints
