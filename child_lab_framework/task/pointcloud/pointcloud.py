import numpy as np
from ...typing.array import IntArray2, FloatArray2, FloatArray4, PointCloud


def project_to_3D(
    fx: float,
    fy: float,
    depth: FloatArray2,
    cx: float,
    cy: float,
    u: IntArray2,
    v: IntArray2,
) -> np.ndarray:
    """
    Converts 2D image coordinates to 3D world coordinates.

    Args:
        fx (float): Focal length along the x-axis.
        fy (float): Focal length along the y-axis.
        depth (FloatArray2): Depth map.
        cx (float): Principal point offset in x direction.
        cy (float): Principal point offset in y direction.
        u (IntArray2): Grid of pixel coordinates along the width (x-coordinates).
        v (IntArray2): Grid of pixel coordinates along the height (y-coordinates).

    Returns:
        FloatArray3: 3D coordinates (x, y, z) as a concatenated array.
    """
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Expand dimensions to make it 3D
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    z = np.expand_dims(z, axis=-1)

    return np.concatenate((x, y, z), axis=-1)


def make_point_cloud(
    rgbd: FloatArray4, fx: float, fy: float, cx: float, cy: float
) -> PointCloud:
    """
    Generates a point cloud from an RGB-D image.

    Args:
        rgbd (FloatArray4): The input RGB-D image, where the last channel is the depth.
        fx (float): Focal length along the x-axis.
        fy (float): Focal length along the y-axis.
        cx (float): Principal point offset in x direction.
        cy (float): Principal point offset in y direction.

    Returns:
        np.ndarray: A point cloud where each point has (x, y, z, r, g, b) values.
    """
    H, W = rgbd.shape[:2]  # Get height and width from the RGBD image dimensions
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    # Extract the depth and RGB channels
    xyz = project_to_3D(
        fx, fy, rgbd[:, :, 3], cx, cy, u, v
    )  # Using depth channel (3rd index)
    rgb = rgbd[:, :, :3]  # Extract the RGB channels

    # Concatenate the xyz coordinates with the RGB values
    point_cloud = np.concatenate((xyz, rgb), axis=-1)

    return point_cloud
