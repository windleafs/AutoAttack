import torch
import torch.nn.functional as F
import random

import os
import cv2
import numpy as np

class AutoAttack(torch.nn.Module):
    def __init__(self, kernel_size=3,sigmaX=1.0,sigmaY=1.0, brightness=0.1, contrast=0.1,erase_prob=0.1,max_area=0.02,max_aspect_ratio=3.0):
        super(AutoAttack, self).__init__()
        self.kernel_size = kernel_size          #高斯核大小
        self.sigmaX = sigmaX                    #X向标准差
        self.sigmaY = sigmaY                    #Y向标准差                    
        self.erase_prob = erase_prob            #擦除概率
        self.max_area = max_area                #最大擦除面积 
        self.max_aspect_ratio=max_aspect_ratio  #最大擦除宽高比
        self.brightness = brightness            #亮度增强值
        self.contrast = contrast                #对比度增强倍数

    def forward(self, x):
        # 随机选择一种攻击方式
        attack_type = random.choice(['gaussian_blur', 'brightness', 'contrast','random_erase'])

        # 高斯模糊
        if attack_type == 'gaussian_blur':
            x = self.apply_gaussian_blur(x,self.kernel_size,sigma_x=self.sigmaX,sigma_y=self.sigmaY)

        # 亮度增加
        elif attack_type == 'brightness':
            x = torch.clamp(x + self.brightness, 0, 1)

        # 对比度增加
        elif attack_type == 'contrast':
            mean = torch.mean(x, dim=(2, 3), keepdim=True)
            x = (x - mean) * (1 + self.contrast) + mean
            x = torch.clamp(x, 0, 1)

        elif attack_type == 'random_erase':
            x=self.random_erase(x,p=self.erase_prob,max_area=self.max_area,max_aspect_ratio=self.max_aspect_ratio)

        return x

    def apply_gaussian_blur(self,x, kernel_size, sigma_x, sigma_y):
        assert len(x.shape) == 4, "Input tensor must have 4 dimensions (batch, channels, height, width)"
        batch_size, channels, height, width = x.shape

        # 计算高斯核
        ksize_x = max(int(kernel_size * sigma_x + 0.5), 1) | 1
        ksize_y = max(int(kernel_size * sigma_y + 0.5), 1) | 1
        kernel_x = torch.exp(-(torch.arange(ksize_x, dtype=torch.float32) - ksize_x // 2) ** 2 / (2 * sigma_x ** 2))
        kernel_y = torch.exp(-(torch.arange(ksize_y, dtype=torch.float32) - ksize_y // 2) ** 2 / (2 * sigma_y ** 2))
        kernel = torch.outer(kernel_x, kernel_y)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, ksize_x, ksize_y).repeat(channels, 1, 1, 1).to(x.device)

        # 使用卷积实现高斯模糊
        padding_x = (ksize_x - 1) // 2
        padding_y = (ksize_y - 1) // 2
        x = F.pad(x, [padding_y, padding_y, padding_x, padding_x], mode='reflect')
        x = F.conv2d(x, kernel, stride=1, groups=channels)

        # 调整输出大小
        if ksize_x != height or ksize_y != width:
            x = x[:, :, :height, :width]

        return x

    def random_erase(self,x, p=0.1, max_area=0.02, max_aspect_ratio=3.0):
        assert len(x.shape) == 4, "Input tensor must have 4 dimensions (batch, channels, height, width)"
        batch_size, channels, height, width = x.shape

        for i in range(batch_size):
            if random.uniform(0, 1) < p:
                # 随机生成擦除区域
                area = random.uniform(0, max_area) * height * width
                aspect_ratio = random.uniform(1.0, max_aspect_ratio)
                h = int(round(math.sqrt(area * aspect_ratio)))
                w = int(round(math.sqrt(area / aspect_ratio)))
                if h < height and w < width:
                    top = random.randint(0, height - h)
                    left = random.randint(0, width - w)
                    bottom = top + h
                    right = left + w
                    x[i, :, top:bottom, left:right] = torch.randn(channels, h, w)

        return x

    




