import asyncio
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat, starmap

from child_lab_framework.core.sequence import imputed_with_reference_inplace
from child_lab_framework.task import face
from ...core.video import Frame
from ...typing.stream import Fiber

type Input = tuple[
    list[Frame | None] | None,
    list[face.Result | None] | None,
]

class Result:
    n_detections: int
    emotions: list[float]

    def __init__(self, n_detections: int, emotions: list[float]) -> None:
        self.n_detections = n_detections
        self.emotions = emotions

class Estimator:
    executor: ThreadPoolExecutor

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.executor = executor

    def predict(self, frame: Frame, faces: face.Result | None) -> Result:
        n_detections = 0
        face_emotions = []
        for face in faces.boxes:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = score_emotions(analysis[0])
            n_detections += 1
            face_emotions.append(emotion)
        
        return Result(n_detections, face_emotions)
    
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
                                zip(
                                    frames,
                                    faces or repeat(None)
                                ),
                            )
                        ),
                    )

                case _:
                    results = None

def score_emotions(emotions):
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