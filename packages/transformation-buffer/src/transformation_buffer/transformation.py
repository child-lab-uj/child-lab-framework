from dataclasses import dataclass
from typing import Literal, Self

import torch
from jaxtyping import Float
from serde import field, serde


def _serialize_tensor(tensor: torch.Tensor) -> str:
    return f'torch.tensor({tensor.tolist()}, dtype={tensor.dtype})'


def _deserialize_tensor(input: str) -> torch.Tensor:
    import torch as __torch

    result = eval(input, {}, {'torch': __torch})
    del __torch

    return result  # type: ignore[no-any-return]


type Axis = Literal['x', 'y', 'z']


def rotation_around(
    axis: Axis,
    angle: float | torch.Tensor,
) -> Float[torch.Tensor, '3 3']:
    a = torch.tensor(angle)
    sin = torch.sin(a)
    cos = torch.cos(a)

    match axis:
        case 'x':
            return torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cos, -sin],
                    [0.0, sin, cos],
                ]
            )

        case 'y':
            return torch.tensor(
                [
                    [cos, 0.0, sin],
                    [0.0, 1.0, 0.0],
                    [-sin, 0.0, cos],
                ]
            )

        case 'z':
            return torch.tensor(
                [
                    [cos, -sin, 0.0],
                    [sin, cos, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )


def translation_along(axis: Axis, distance: float) -> Float[torch.Tensor, '3']:
    t = torch.zeros(3)

    match axis:
        case 'x':
            t[0] = distance
        case 'y':
            t[1] = distance
        case 'z':
            t[2] = distance

    return t


@serde
@dataclass(eq=False)
class Transformation:
    rotation_and_translation: Float[torch.Tensor, '4 4'] = field(
        serializer=_serialize_tensor,
        deserializer=_deserialize_tensor,
    )

    def __init__(
        self,
        rotation_and_translation: Float[torch.Tensor, '4 4'],
    ) -> None:
        self.rotation_and_translation = rotation_and_translation.clone()

    def __eq__(self, other: object, /) -> bool:
        return isinstance(other, Transformation) and bool(
            self.rotation_and_translation.eq(other.rotation_and_translation).all()
        )

    def __matmul__(self, other: 'Transformation') -> 'Transformation':
        return Transformation(
            other.rotation_and_translation @ self.rotation_and_translation
        )

    def inverse(self) -> 'Transformation':
        return Transformation(self.rotation_and_translation.inverse())

    @staticmethod
    def approx_eq(
        t1: 'Transformation',
        t2: 'Transformation',
        absolute_tolerance: float = 1e-8,
    ) -> bool:
        return bool(
            torch.isclose(
                t1.rotation_and_translation,
                t2.rotation_and_translation,
                atol=absolute_tolerance,
            ).all()
        )

    # Cannot write `IDENTITY: ClassVar[Self] = Transformation.from_parts`.
    # Class properties are also deprecated.
    # Python... :v
    @classmethod
    def identity(cls) -> Self:
        return cls.from_parts(
            torch.eye(3, 3, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
        )

    @classmethod
    def from_parts(
        cls,
        rotation: Float[torch.Tensor, '3 3'],
        translation: Float[torch.Tensor, '3'],
    ) -> Self:
        rotation_and_translation = torch.zeros(
            (4, 4),
            dtype=rotation.dtype,
            device=rotation.device,
        )
        rotation_and_translation[:3, :3] = rotation
        rotation_and_translation[:3, 3] = translation
        rotation_and_translation[3, 3] = 1.0

        return cls(rotation_and_translation)

    @classmethod
    def active(
        cls,
        intrinsic_euler_angles: Float[torch.Tensor, '3'] | None = None,
        translation: Float[torch.Tensor, '3'] | None = None,
    ) -> Self:
        rotation = (
            (
                rotation_around('x', intrinsic_euler_angles[0])
                @ rotation_around('y', intrinsic_euler_angles[1])
                @ rotation_around('z', intrinsic_euler_angles[2])
            )
            if intrinsic_euler_angles is not None
            else torch.eye(3)
        )

        translation = translation if translation is not None else torch.zeros(3)

        return cls.from_parts(rotation, translation)

    def clone(self) -> 'Transformation':
        return Transformation(self.rotation_and_translation.clone())

    def cpu(self) -> 'Transformation':
        return Transformation(self.rotation_and_translation.cpu())

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> 'Transformation':
        return Transformation(self.rotation_and_translation.to(device, dtype, copy=True))
