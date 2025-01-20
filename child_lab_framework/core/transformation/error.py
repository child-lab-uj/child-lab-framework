import numpy as np

from .interface import ProjectableAndTransformable, Unprojectable
from .transformation import Transformation


def reprojection_error[T: ProjectableAndTransformable[Unprojectable[object]]](
    object_in_a: T,
    object_in_b: T,
    a_to_b: Transformation,
) -> float:
    true_points_in_b = object_in_b.flat_points
    projected_points_in_b = object_in_a.transform(a_to_b).flat_points

    return float(np.linalg.norm(true_points_in_b - projected_points_in_b))
