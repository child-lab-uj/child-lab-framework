import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Self

import torch
from child_lab_data.io.point_cloud import Reader as PointCloudReader
from child_lab_data.io.result_tensor import Writer as ResultWriter
from transformation_buffer.buffer import Buffer
from video_io.calibration import Calibration
from video_io.reader import Reader as VideoReader
from video_io.writer import Writer as VideoWriter
from vpc import face, gaze, pose

from child_lab_procedures.support.imputation import (
    imputed_with_closest_known_reference as imputed,
)

__all__ = ['Configuration', 'Procedure', 'VideoIoContext', 'VisualizationContext']


@dataclass(slots=True)
class Configuration:
    device: torch.device
    batch_size: int
    max_detections: int

    yolo_checkpoint: Path
    mini_face_models_directory: Path

    face_confidence_threshold: float
    face_suppression_threshold: float


class VisualizationContext(
    pose.VisualizationContext,
    face.VisualizationContext,
    gaze.VisualizationContext,
): ...


@dataclass(slots=True)
class VideoIoContext:
    name: str
    calibration: Calibration

    video_reader: VideoReader
    point_cloud_reader: PointCloudReader

    video_writer: VideoWriter[VisualizationContext] | None = None
    pose_writer: ResultWriter[pose.Result3d] | None = None
    gaze_writer: ResultWriter[gaze.Result3d] | None = None


@dataclass(slots=True)
class VideoAnalysisContext:
    name: str
    calibration: Calibration

    video_reader: VideoReader
    point_cloud_reader: PointCloudReader

    pose_estimator: pose.Estimator
    face_estimator: face.Estimator
    gaze_estimator: gaze.Estimator

    video_writer: VideoWriter[VisualizationContext] | None = None
    pose_writer: ResultWriter[pose.Result3d] | None = None
    gaze_writer: ResultWriter[gaze.Result3d] | None = None

    @classmethod
    def from_io(cls, io: VideoIoContext, configuration: Configuration) -> Self:
        pose_estimator = pose.Estimator(
            configuration.max_detections,
            configuration.yolo_checkpoint,
            configuration.device,
        )
        face_estimator = face.Estimator(
            configuration.face_confidence_threshold,
            configuration.face_suppression_threshold,
            configuration.device,
        )
        gaze_estimator = gaze.Estimator(
            io.calibration,
            configuration.mini_face_models_directory,
            fps=int(io.video_reader.metadata.fps),
        )

        return cls(
            io.name,
            io.calibration,
            io.video_reader,
            io.point_cloud_reader,
            pose_estimator,
            face_estimator,
            gaze_estimator,
            io.video_writer,
            io.pose_writer,
            io.gaze_writer,
        )


class Procedure:
    """
    Computes **all** possible features eagerly.
    """

    contexts: list[VideoAnalysisContext]
    transformation_buffer: Buffer[str]

    configuration: Configuration

    def __init__(
        self,
        configuration: Configuration,
        io_contexts: list[VideoIoContext],
        transformation_buffer: Buffer[str],
    ) -> None:
        self.configuration = configuration
        self.contexts = [
            VideoAnalysisContext.from_io(io, configuration) for io in io_contexts
        ]
        self.transformation_buffer = transformation_buffer

    def run(self, on_step: Callable[[], Any] | None = None) -> None:
        while True:
            if on_step is not None:
                on_step()

            exhausted = True

            for context in self.contexts:
                name = context.name

                frame_batch = context.video_reader.read_batch(
                    self.configuration.batch_size
                )
                if frame_batch is None:
                    logging.debug(f'"{name}" - No frames read')
                    continue

                logging.debug(f'"{name}" - {frame_batch.shape[0]} frames read')

                n_frames = len(frame_batch)
                exhausted = False

                point_clouds = imputed(
                    [context.point_cloud_reader.read() for _ in range(n_frames)]
                )
                logging.debug(f'"{name}" - Point clouds loaded')

                poses = imputed(context.pose_estimator.predict_batch(frame_batch))
                logging.debug(f'"{name}" - Poses detected')

                unprojected_poses = (
                    [
                        pose.unproject(point_cloud)
                        for pose, point_cloud in zip(poses, point_clouds)
                    ]
                    if poses is not None and point_clouds is not None
                    else None
                )
                logging.debug(f'"{name}" - Poses unprojected')

                faces = (
                    imputed(context.face_estimator.predict_batch(frame_batch, poses))
                    if poses is not None
                    else None
                )
                logging.debug(f'"{name}" - Faces detected')

                gazes = (
                    imputed(context.gaze_estimator.predict_batch(frame_batch, faces))
                    if faces is not None
                    else None
                )
                logging.debug(f'"{name}" - Gaze detected')

                if gazes is not None and unprojected_poses is not None:
                    gazes = [
                        gaze.align(context.calibration, pose.cpu())
                        for gaze, pose in zip(gazes, unprojected_poses)
                    ]
                    logging.debug(f'"{name}" - Gaze corrected')

                projected_gazes = (
                    [gaze.project(context.calibration) for gaze in gazes]
                    if gazes is not None
                    else None
                )
                logging.debug(f'"{name}" - Gaze projected')

                if context.video_writer is not None:
                    items_to_draw = [
                        filter(None, frame_items)
                        for frame_items in zip(
                            poses or repeat(None),
                            faces or repeat(None),
                            projected_gazes or repeat(None),
                        )
                    ]
                    context.video_writer.write_batch(frame_batch, items_to_draw)
                    logging.debug(f'"{name}" - Output video batch saved')

                if context.gaze_writer is not None:
                    if gazes is None:
                        context.gaze_writer.skip(n_frames)
                    else:
                        for result in gazes:
                            context.gaze_writer.write(result)

                if context.pose_writer is not None:
                    if unprojected_poses is None:
                        context.pose_writer.skip(n_frames)
                    else:
                        for pose in unprojected_poses:
                            context.pose_writer.write(pose)

            if exhausted:
                break
