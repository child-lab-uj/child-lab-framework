import torch
from beartype import beartype
from jaxtyping import Float
from plum import dispatch, overload
from vpc import gaze, pose


@beartype
@overload
def serialize(item: pose.Result3d) -> Float[torch.Tensor, 'n_detections 74']:  # noqa: F811
    flat_boxes = item.boxes  # n_detections x 6
    flat_keypoints = item.keypoints.flatten(1, -1)  # n_detections x 17 * 4
    return torch.cat((flat_boxes, flat_keypoints), dim=1)


@beartype
@overload
def deserialize(  # noqa: F811
    ty: type[pose.Result3d],
    item: Float[torch.Tensor, 'n_detections 74'],
) -> pose.Result3d:
    boxes = item[:, :6]
    keypoints = item[:, 6:].unflatten(-1, (17, 4))
    return pose.Result3d(boxes, keypoints, item.device)


@beartype
@overload
def serialize(item: pose.Result) -> Float[torch.Tensor, 'n_detections 56']:  # noqa: F811
    flat_boxes = item.boxes  # n_detections x 5
    flat_keypoints = item.keypoints.flatten(1, -1)  # n_detections x 17 * 3
    return torch.cat((flat_boxes, flat_keypoints), dim=1)


@beartype
@overload
def deserialize(  # noqa: F811
    ty: type[pose.Result],
    item: Float[torch.Tensor, 'n_detections 56'],
) -> pose.Result:
    boxes = item[:, :6, :]
    keypoints = item[:, 6:, :].unflatten(-1, (17, 3))
    return pose.Result(boxes, keypoints, item.device)


@beartype
@overload
def serialize(item: gaze.Result3d) -> Float[torch.Tensor, 'n_detections 4 3']:  # noqa: F811
    return torch.cat((item.eyes, item.directions), dim=1)


@beartype
@overload
def deserialize(  # noqa: F811
    ty: type[gaze.Result3d],
    item: Float[torch.Tensor, 'n_detections 4 3'],
) -> gaze.Result3d:
    eyes = item[:, :2, :]
    directions = item[:, 2:, :]
    return gaze.Result3d(eyes, directions, item.device)


@dispatch
def serialize(item: object) -> torch.Tensor: ...  # noqa: F811


@dispatch
def deserialize(ty: type, item: object) -> object: ...  # noqa: F811
