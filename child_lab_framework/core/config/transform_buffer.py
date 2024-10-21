import yaml
import numpy as np
import pytransform3d.transformations as pt
from pytransform3d.transform_manager import TransformManager
from threading import Lock
from dataclasses import dataclass, field

from ...typing.array import FloatArray2, FloatArray1


class TransformBuffer:
    @dataclass
    class Transform:
        input_frame: str
        output_frame: str
        rotation: FloatArray2
        translation: FloatArray1

        def to_dict(self):
            # Convert the numpy arrays to regular lists for YAML compatibility
            return {
                'input_frame': self.input_frame,
                'output_frame': self.output_frame,
                'rotation': self.rotation.tolist(),
                'translation': self.translation.tolist(),
            }

        @classmethod
        def from_dict(cls, data):
            return cls(
                input_frame=data['input_frame'],
                output_frame=data['output_frame'],
                rotation=np.array(data['rotation']),
                translation=np.array(data['translation']),
            )

    @dataclass
    class State:
        transforms: list['TransformBuffer.Transform'] = field(default_factory=list)

        def save_to_yaml(self, file_path: str):
            data = {'transforms': [t.to_dict() for t in self.transforms]}

            with open(file_path, 'w') as yaml_file:
                yaml.dump(data, yaml_file)

        @classmethod
        def load_from_yaml(cls, file_path: str) -> 'TransformBuffer.State':
            with open(file_path, 'r') as yaml_file:
                data = yaml.load(yaml_file, Loader=yaml.SafeLoader)

            return cls(
                [TransformBuffer.Transform.from_dict(t) for t in data['transforms']]
            )

    def __init__(self):
        self.__tm = TransformManager()
        self.__mutex = Lock()

    @classmethod
    def from_state(cls, state: State):
        tf_buffer = cls()
        for transform in state.transforms:
            tf_buffer.add_transform(
                transform.input_frame,
                transform.output_frame,
                transform.rotation,
                transform.translation,
            )
        return tf_buffer

    def add_transform(
        self,
        input_frame: str,
        output_frame: str,
        rotation: FloatArray2,
        translation: FloatArray1,
    ):
        with self.__mutex:
            self.__tm.add_transform(
                input_frame,
                output_frame,
                pt.transform_from(
                    R=rotation,
                    p=translation,
                ),
            )

    def get_transform(
        self, input_frame: str, output_frame: str
    ) -> tuple[FloatArray2, FloatArray1]:
        with self.__mutex:
            tf = self.__tm.get_transform(input_frame, output_frame)

        rotation = tf[:3, :3]
        translation = tf[:3, 3]

        return rotation, translation

    def transform(self, input_frame: str, output_frame: str, point: FloatArray2):
        rotation, translation = self.get_transform(
            input_frame, output_frame
        )
        return (rotation @ point) + translation

    def get_state(self) -> State:
        state = self.State()
        for input_frame, output_frame in self.__tm.transforms:
            rotation, translation = self.get_transform(
                input_frame, output_frame
            )

            state.transforms.append(
                self.Transform(
                    input_frame=input_frame,
                    output_frame=output_frame,
                    rotation=rotation,
                    translation=translation,
                )
            )
        return state


if __name__ == '__main__':
    tf_buffer = TransformBuffer()

    R = [
        [0.7152960, 0.0369892, 0.6978420],
        [0.3929953, 0.8044356, -0.4454638],
        [-0.5778463, 0.5928871, 0.5608730],
    ]

    p = np.array([26, 21, 34])

    obs = tf_buffer.add_transform(
        input_frame='root', output_frame='test', rotation=R, translation=p
    )

    state = tf_buffer.get_state()
    state.save_to_yaml('/tmp/test.yaml')
    state2 = TransformBuffer.State.load_from_yaml('/tmp/test.yaml')
    tf_buffer = TransformBuffer.from_state(state2)

    point = np.array([12, 26, 13])
    print('original:', *point, sep='\n', end='\n\n')
    transformed = tf_buffer.transform('root', 'test', point)
    print('root -> test: ', *transformed, sep='\n', end='\n\n')
    transformed = tf_buffer.transform('test', 'root', transformed)
    print('test -> root: ', *transformed, sep='\n', end='\n\n')
