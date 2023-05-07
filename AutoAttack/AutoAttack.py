import torch
import torch.nn.functional as F
import random

import os
import cv2
import numpy as np

class AutoAttack(torch.nn.Module):
    def __init__(self, sigma=1.0, brightness=0.1, contrast=0.1):
        super(AutoAttack, self).__init__()
        self.sigma = sigma
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, x):
        # 随机选择一种攻击方式
        attack_type = random.choice(['gaussian_blur', 'brightness', 'contrast'])

        # 高斯模糊
        if attack_type == 'gaussian_blur':
            x = self.apply_gaussian_blur(x, sigma=self.sigma)

        # 亮度增加
        elif attack_type == 'brightness':
            x = torch.clamp(x + self.brightness, 0, 1)

        # 对比度增加
        elif attack_type == 'contrast':
            mean = torch.mean(x, dim=(2, 3), keepdim=True)
            x = (x - mean) * (1 + self.contrast) + mean
            x = torch.clamp(x, 0, 1)

        return x

    def apply_gaussian_blur(self, x, sigma):
        # 计算高斯核
        ksize = int(sigma * 3) * 2 + 1
        kernel = torch.tensor([[i - ksize // 2 + 0.5, j - ksize // 2 + 0.5] for i in range(ksize) for j in range(ksize)], dtype=torch.float32)
        kernel = torch.exp(-(kernel[:, 0] ** 2 + kernel[:, 1] ** 2) / (2 * sigma ** 2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, ksize, ksize).repeat(x.shape[1], 1, 1, 1).to(x.device)

        # 使用卷积实现高斯模糊
        x = F.pad(x, [ksize // 2, ksize // 2, ksize // 2, ksize // 2], mode='reflect')
        x = F.conv2d(x, kernel, stride=1, groups=x.shape[1])

        return x
        





