import asyncio
from typing import Dict, List
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat, starmap

from child_lab_framework.core.sequence import imputed_with_reference_inplace
from child_lab_framework.task import face
from ...core.video import Frame
from ...typing.stream import Fiber
from ...typing.array import FloatArray2

type Input = tuple[
    list[Frame | None] | None,
    list[face.Result | None] | None,
]


class Result:
    emotions: list[float]
    boxes: list[FloatArray2]

    def __init__(self, emotions: list[float], boxes: list[FloatArray2]) -> None:
        self.emotions = emotions
        self.boxes = boxes


class Estimator:
    executor: ThreadPoolExecutor

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.executor = executor

    def predict(self, frame: Frame, faces: face.Result | None) -> Result:
        face_emotions = []
        boxes = []
        frame_height, frame_width, _ = frame.shape
        for face_box in faces.boxes:
            x_min, y_min, x_max, y_max = face_box
            x_min = max(x_min - 50, 0)
            x_max = min(x_max + 50, frame_width)
            y_min = max(y_min - 50, 0)
            y_max = min(y_max + 50, frame_height)
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            analysis = DeepFace.analyze(
                cropped_frame, actions=['emotion'], enforce_detection=False
            )
            emotion = score_emotions(analysis[0])
            face_emotions.append(emotion)
            boxes.append(face_box)

        return Result(face_emotions, boxes)

    def predict_batch(
        self,
        frames: list[Frame],
        faces: list[face.Result | None],
    ) -> list[Result] | None:
        return imputed_with_reference_inplace(
            list(starmap(self.predict, zip(frames, faces)))
        )

    async def stream(
        self,
    ) -> Fiber[list[Frame | None] | None, list[Result | None] | None]:
        loop = asyncio.get_running_loop()
        executor = self.executor

        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case (
                    list(frames),
                    faces,
                ):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: list(
                            starmap(
                                self.__predict,
                                zip(frames, faces or repeat(None)),
                            )
                        ),
                    )

                case _:
                    results = None


def score_emotions(emotions: List[Dict[str, float]]) -> float:
    # Most of the time, "angry" and "fear" are similar to "neutral" in the reality
    scores = {
        'angry': -0.05,
        'disgust': 0,
        'fear': -0.07,
        'happy': 1,
        'sad': -1,
        'surprise': 0,
        'neutral': 0,
    }
    val = 0
    for emotion, score in scores.items():
        val += emotions['emotion'][emotion] * score

    return val
