import torch
from jaxtyping import Float


def rotation_matrix_between_vectors(
    from_vector: Float[torch.Tensor, '3'],
    to_vector: Float[torch.Tensor, '3'],
) -> Float[torch.Tensor, '3 3'] | None:
    """
    Finds a `3 x 3` rotation matrix `R` such that `R @ from_vector = to_vector`.
    """

    from_normalized = from_vector / from_vector.norm(2.0)
    to_normalized = to_vector / to_vector.norm(2.0)

    cross_product = torch.cross(from_normalized, to_normalized)

    rotation_angle_sine = cross_product.norm(2.0)
    rotation_angle_cosine = torch.dot(from_normalized, to_normalized)

    rotation_axis = cross_product / rotation_angle_sine
    axis_matrix = torch.outer(rotation_axis, rotation_axis)
    axis_cross_product_matrix = cross_product_matrix(rotation_axis)

    identity = torch.eye(3, dtype=torch.float32)

    rotation = (
        rotation_angle_cosine * identity
        + (1.0 - rotation_angle_cosine) * axis_matrix
        + rotation_angle_sine * axis_cross_product_matrix
    )

    if not rotation.isfinite().all():
        return None

    return rotation


def cross_product_matrix(vector: Float[torch.Tensor, '3']) -> Float[torch.Tensor, '3 3']:
    """
    Compute a skew-symmetric `3 x 3` matrix `C` such that for any vector `v`,
    the cross product of `vector` and `v` is equal to `C @ v`.
    """

    x, y, z = vector

    return torch.tensor(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=torch.float32,
    )
