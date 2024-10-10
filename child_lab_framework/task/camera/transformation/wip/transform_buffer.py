import numpy as np
import pytransform3d.transformations as pt
from pytransform3d.transform_manager import TransformManager


class TransformBuffer:
    def __init__(self):
        self._tm = TransformManager()

    def add_transform(
        self,
        input_frame: str,
        output_frame: str,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
    ):
        self._tm.add_transform(
            input_frame,
            output_frame,
            pt.transform_from(
                R=rotation_matrix,
                p=translation_vector,
            ),
        )

    def get_transform(
        self, input_frame: str, output_frame: str
    ) -> tuple[np.ndarray, np.ndarray]:
        tf = self._tm.get_transform(input_frame, output_frame)

        rotation_matrix = tf[:3, :3]
        translation_vector = tf[:3, 3]

        return rotation_matrix, translation_vector

    def transform(self, input_frame: str, output_frame: str, point: np.ndarray):
        rotation_matrix, translation_vector = self.get_transform(
            input_frame, output_frame
        )
        return (rotation_matrix @ point) + translation_vector


if __name__ == '__main__':
    tf_buffer = TransformBuffer()

    R = [
        [0.7152960, 0.0369892, 0.6978420],
        [0.3929953, 0.8044356, -0.4454638],
        [-0.5778463, 0.5928871, 0.5608730],
    ]

    p = np.array([26, 21, 34])

    obs = tf_buffer.add_transform(
        input_frame='root', output_frame='test', rotation_matrix=R, translation_vector=p
    )

    point = np.array([12, 26, 13])
    print('original:', *point, sep='\n', end='\n\n')
    transformed = tf_buffer.transform('root', 'test', point)
    print('root -> test: ', *transformed, sep='\n', end='\n\n')
    transformed = tf_buffer.transform('test', 'root', transformed)
    print('test -> root: ', *transformed, sep='\n', end='\n\n')
