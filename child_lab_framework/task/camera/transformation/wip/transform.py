def rigid_transform_3D(points_input_frame: np.ndarray, points_output_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert points_input_frame.shape == points_output_frame.shape

    num_rows, num_cols = points_input_frame.shape
    if num_rows != 3:
        raise Exception(f"matrix points_input_frame is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = points_output_frame.shape
    if num_rows != 3:
        raise Exception(f"matrix points_output_frame is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(points_input_frame, axis=1)
    centroid_B = np.mean(points_output_frame, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = points_input_frame - centroid_A
    Bm = points_output_frame - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < 0, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t.squeeze()