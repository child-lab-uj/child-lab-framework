import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from pyntcloud import PyntCloud


class Metric3D:
    def __init__(self, onnx_path, camera_matrix, providers=None):
        # providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '0', 'device_id': str(gpu_index)})]
        if providers is None:
            providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        self.ort_session = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=sess_options
        )
        input_shape = self.ort_session.get_inputs()[0].shape  # [1, 3, h, w]
        self.inference_h = input_shape[2]
        self.inference_w = input_shape[3]

        self.camera_matrix = np.zeros((3, 4))
        self.camera_matrix[0:3, 0:3] = np.array(camera_matrix)

    def resize(self, image, camera_matrix=None):
        self.h0, self.w0 = image.shape[0:2]
        scale = min(self.inference_h / self.h0, self.inference_w / self.w0)
        self.scale = scale
        self.h_eff = int(self.h0 * scale)
        self.w_eff = int(self.w0 * scale)
        final_image = np.zeros([self.inference_h, self.inference_w, 3])
        final_image[0 : self.h_eff, 0 : self.w_eff] = cv2.resize(
            image, (self.w_eff, self.h_eff), interpolation=cv2.INTER_LINEAR
        )
        if camera_matrix is None:
            return final_image
        else:
            camera_matrix = camera_matrix.copy()
            camera_matrix[0:2, :] = camera_matrix[0:2, :] * scale
            return final_image, camera_matrix

    def deresize(self, seg):
        seg = seg[0 : self.h_eff, 0 : self.w_eff]
        seg = cv2.resize(seg, (self.w0, self.h0), interpolation=cv2.INTER_NEAREST)
        return seg

    def join(self):
        threading.Thread.join(self)
        threading.Thread.__init__(self)
        return self._output

    def run(self, image_rgb):
        start_time = time.time()
        # Prepare input for model inference
        onnx_input, pad_info = self.prepare_input(image_rgb)
        # Perform inference
        outputs = self.ort_session.run(None, onnx_input)
        depth_image = outputs[0][0, 0]  # [1, 1, H, W] -> [H, W]
        point_cloud = outputs[1]  # [HW, 6]
        mask = outputs[2]  # [HW]

        point_cloud = point_cloud[mask]  # [HW, 6]
        point_cloud = point_cloud.reshape([-1, 6])
        point_cloud[:, 3:] *= 255  # Map colors to 0-255 range

        depth_image = depth_image[
            pad_info[0] : depth_image.shape[0] - pad_info[1],
            pad_info[2] : depth_image.shape[1] - pad_info[3],
        ]  # [H, W] -> [h, w]

        print(f'Metric3D runtime: {time.time() - start_time}')
        return depth_image, point_cloud

    def prepare_input(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, list[int]]:
        input_size = (616, 1064)

        h, w = image_rgb.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        self.scale = scale
        rgb = cv2.resize(
            image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )

        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb: np.ndarray = cv2.copyMakeBorder(
            rgb,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding,
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        camera_matrix = self.camera_matrix.copy()
        camera_matrix[0:2, :] = camera_matrix[0:2, :] * scale

        # Create camera_matrix_inv
        camera_matrix_expanded = np.eye(4)
        camera_matrix_expanded[0:3, 0:4] = camera_matrix
        camera_matrix_inv = np.linalg.inv(camera_matrix_expanded)  # 4x4

        # Create T
        T = np.eye(4)

        # Create mask
        H, W = input_size
        mask = np.zeros([H, W], dtype=np.uint8)
        mask[pad_info[0] : H - pad_info[1], pad_info[2] : W - pad_info[3]] = 1

        onnx_input = {
            'image': np.ascontiguousarray(
                np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32
            ),  # 1, 3, H, W
            'P': camera_matrix.astype(np.float32)[None],  # 1, 3, 4
            'P_inv': camera_matrix_inv.astype(np.float32)[None],  # 1, 4, 4
            'T': T.astype(np.float32)[None],  # 1, 4, 4
            'mask': mask.astype(bool)[None],  # 1, H, W
        }

        return onnx_input, pad_info


if __name__ == '__main__':
    camera_matrix = np.array(
        [
            [524.86812468, 0.0, 524.92448944],
            [0.0, 352.13927069, 707.66841893],
            [0.0, 0.0, 1.0],
        ]
    )

    m3d = Metric3D(
        onnx_path='/Users/igor/Work/child-lab/child-lab-environment/models/metric_3d.onnx',
        camera_matrix=camera_matrix,
    )

    img = cv2.imread(
        '/Users/igor/Work/child-lab/depth_estimation/Metric3D/images/janek.webp'
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = m3d.run(img)

    depth_image, pointcloud = output

    # print(pointcloud.shape)

    # cv2.imshow('image', depth_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    pointcloud = pointcloud[::10]

    df = pd.DataFrame(pointcloud, columns=['x', 'y', 'z', 'blue', 'green', 'red'])
    print(df.head())

    cloud = PyntCloud(df)
    cloud.plot(initial_point_size=10)
