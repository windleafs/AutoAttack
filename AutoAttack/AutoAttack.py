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
        


def grayscale(img):
    # 将彩色图像转换为灰度图像
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def random_mask(img):
    # 随机生成遮挡区域
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
    x2, y2 = random.randint(w // 2, w), random.randint(h // 2, h)
    cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
    return mask

def random_flip(img):
    # 随机水平翻转或垂直翻转图像
    flip_code = random.choice([-1, 0, 1])
    return cv2.flip(img, flip_code)

def apply_transforms(img):
    # 将图像随机转换为灰度图、随机遮挡和随机翻转
    img = grayscale(img)
    mask = random_mask(img)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = random_flip(img)
    return img

def process_images_in_directory(input_dir, output_dir):
    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 读取图片并进行处理
            input_path = os.path.join(input_dir, filename)
            img = cv2.imread(input_path)
            img = apply_transforms(img)

            # 将处理后的图片保存到输出目录
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)


