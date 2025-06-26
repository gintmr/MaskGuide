# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        
        original_size = > H,W
        coords = > W,H
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(self, coords: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        # 确保 coords 是 torch.Tensor 类型
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float)

        # 确保 coords 是浮点类型
        coords = coords.to(dtype=torch.float)

        # 提取原始图像的高和宽
        old_h, old_w = original_size

        # 计算新的图像尺寸
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_length)

        # # 打印调试信息
        # print(f"Original size: (H={old_h}, W={old_w})")
        # print(f"New size: (H={new_h}, W={new_w})")
        # print(f"coords before scaling: {coords}")

        # 缩放坐标
        coords[..., 0] = coords[..., 0] * (new_w / old_w)  # 缩放 x 坐标
        coords[..., 1] = coords[..., 1] * (new_h / old_h)  # 缩放 y 坐标

        # 打印调试信息
        # print(f"coords after scaling: {coords}")

        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        device = None
        if hasattr(boxes, "device"):
            device = boxes.device
        if isinstance(boxes, torch.Tensor):
            boxes = np.array(boxes.cpu())
        else:
            boxes = np.array(boxes)
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4).to(device) if device is not None else boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        # 将neww和newh四舍五入取整
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
