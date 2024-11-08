import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, canny_weight=1.0, low_threshold=5, high_threshold=20):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.canny_weight = canny_weight
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, pred, target):
        # MSE Loss
        mse_loss = F.mse_loss(pred, target)

        # Canny Edge Loss
        canny_loss = self._compute_canny_loss(pred, target)

        # Combined Loss
        combined_loss = self.mse_weight * mse_loss + self.canny_weight * canny_loss

        return combined_loss

    def _compute_canny_loss(self, y_pred, y_true):
        # 对输入数据进行归一化，使其值范围在0到255之间
        y_true_normalized = y_true / 5000.0 * 255
        y_pred_normalized = y_pred / 5000.0 * 255

        # 将浮点类型的图像转换为8位无符号整数类型
        y_true_uint8 = y_true_normalized.detach().cpu().numpy().astype(np.uint8)
        y_pred_uint8 = y_pred_normalized.detach().cpu().numpy().astype(np.uint8)

        # 对真实标签使用 Canny 边缘检测算法生成边缘图像
        true_edges = [cv2.Canny(image[0], 5, 20) for image in y_true_uint8]



        true_edges = np.stack(true_edges, axis=0)
        true_edges = torch.tensor(true_edges).float().unsqueeze(1) / 255.0  # 归一化到 [0, 1] 范围
        # true_edges = torch.tensor(true_edges) / 255.0

        # 对模型生成的边缘图像使用 Canny 边缘检测算法生成边缘图像
        pred_edges = [cv2.Canny(image[0], 5, 20) for image in y_pred_uint8]

        pred_edges = np.stack(pred_edges, axis=0)
        pred_edges = torch.tensor(pred_edges).float().unsqueeze(1) / 255.0  # 归一化到 [0, 1] 范围

        # 计算两个边缘图像之间的差异作为损失
        loss = F.mse_loss(true_edges, pred_edges, reduction='mean')
        return loss