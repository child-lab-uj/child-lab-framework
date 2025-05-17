import torch
import viser
from child_lab_data.io.result_tensor import Reader
from transformation_buffer.transformation import Transformation
from vpc import gaze, pose

from child_lab_visualization.schema import Frame


def show_vpc_results(
    server: viser.ViserServer,
    origin_name: str,
    transformation: Transformation = Transformation.identity(),
    pose_reader: Reader[pose.Result3d] | None = None,
    gaze_reader: Reader[gaze.Result3d] | None = None,
) -> None:
    uri = Frame(0)

    if pose_reader is not None:
        poses = pose_reader.read()

        if poses is not None:
            segments: list[torch.Tensor] = []  # tensors with shape (2, 3)

            for person_keypoints in poses.keypoints.unbind():
                for start, end in pose.model.YOLO_SKELETON:
                    if (
                        person_keypoints[start, -1] < 0.75
                        or person_keypoints[end, -1] < 0.75
                    ):
                        continue

                    segments.append(person_keypoints[[start, end], :3])

            points = torch.stack(segments).cpu().numpy()

            server.scene.add_line_segments(
                str(uri.pose('collective', origin_name)),
                points=points,
                colors=(0.0, 0.0, 255.0),
                line_width=2,
                wxyz=transformation.rotation_quaternion().numpy(),
                position=transformation.translation().numpy(),
            )

    if gaze_reader is not None:
        gazes = gaze_reader.read()

        if gazes is not None:
            starts = gazes.eyes.flatten(0, 1).unsqueeze(1).cpu()
            directions = gazes.directions.flatten(0, 1).unsqueeze(1).cpu()
            ends = starts + directions

            points = torch.cat((starts, ends), dim=1).numpy()

            server.scene.add_line_segments(
                str(uri.gaze('collective', origin_name)),
                points=points,
                colors=(0.0, 255.0, 0.0),
                line_width=2,
                wxyz=transformation.rotation_quaternion().numpy(),
                position=transformation.translation().numpy(),
            )
