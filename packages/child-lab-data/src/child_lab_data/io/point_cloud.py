import logging
from itertools import count
from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import BFloat16, Float
from tensordict import TensorDict

type PointCloud = Float[torch.Tensor, '3 height width']


class Reader:
    directory: Path
    counter: 'count[int]'

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.counter = count()

    def skip(self, count: int) -> None:
        for _ in range(count):
            next(self.counter)

    @beartype
    def read(self, id: int | None = None) -> PointCloud | None:
        if id is None:
            id = next(self.counter)

        path = self.directory / f'{id}.pt'
        if not path.is_file():
            logging.warning(f'Point cloud not found at {path}')
            return None

        logging.debug(f'Decoding {path}')

        serialized_point_cloud: TensorDict = torch.load(path)

        u: BFloat16[torch.Tensor, '3 bond_dim height']
        s: BFloat16[torch.Tensor, '3 bond_dim']
        vt: BFloat16[torch.Tensor, '3 bond_dim width']

        u = serialized_point_cloud['u']
        s = serialized_point_cloud['s'].squeeze(-1)
        vt = serialized_point_cloud['vt']

        point_cloud: Float[torch.Tensor, '3 height width']
        point_cloud = torch.einsum('cbh, cb, cbw -> chw', u, s, vt).to(torch.float32)
        return point_cloud


class Writer:
    directory: Path
    explained_variance_ratio: float
    counter: 'count[int]'

    def __init__(self, directory: Path, explained_variance_ratio: float = 0.97) -> None:
        self.directory = directory
        self.explained_variance_ratio = explained_variance_ratio
        self.counter = count()

    def skip(self, count: int) -> None:
        for _ in range(count):
            next(self.counter)

    @beartype
    def write(self, point_cloud: PointCloud, id: int | None = None) -> None:
        if id is None:
            id = next(self.counter)

        path = self.directory / f'{id}.pt'
        if path.is_file():
            logging.warning(f'Overwriting the existing point cloud at {path}')

        u, s, v = torch.svd(point_cloud)

        explained_variance = s.div(s.sum(1).reshape(-1, 1)).cumsum(1)

        mask = (
            (explained_variance[0, ...] < self.explained_variance_ratio)
            | (explained_variance[1, ...] < self.explained_variance_ratio)
            | (explained_variance[2, ...] < self.explained_variance_ratio)
        )

        bond_dim = int(mask.sum().item())

        u = u[..., mask].transpose(1, 2)  # (3, bond_dim, height)
        s = s[..., mask].unsqueeze(-1)  # (3, bond_dim)
        vt = v[..., mask].transpose(1, 2)  # (3, bond_dim, width)

        compressed_cloud = TensorDict(
            {'u': u, 's': s, 'vt': vt},
            batch_size=[3, bond_dim],
        ).to(torch.bfloat16)

        torch.save(compressed_cloud, path)
