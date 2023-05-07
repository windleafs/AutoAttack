import os
import cv2
import numpy as np
import random

class ImgEnhance():
    def __init__(self):
        super(ImgEnhance, self).__init__()

    def grayscale(self,img):
        # 将彩色图像转换为灰度图像
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def random_mask(self,img):
        # 随机生成遮挡区域
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
        x2, y2 = random.randint(w // 2, w), random.randint(h // 2, h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
        return mask

    def random_flip(self,img):
        # 随机水平翻转或垂直翻转图像
        flip_code = random.choice([-1, 0, 1])
        return cv2.flip(img, flip_code)

    def apply_transforms(self,img,operation):
        # 将图像随机转换为灰度图、随机遮挡和随机翻转
        if operation=='gray':
            img = self.grayscale(img)
        elif operation=='mask':
            mask = self.random_mask(img)
            img = cv2.bitwise_and(img, img, mask=mask)
        elif operation=='flip':
            img = self.random_flip(img)
        return img

    def process_images_in_directory(self,input_dir, output_dir,operation,prefix=''):
        # 遍历输入目录下的所有文件
        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # 读取图片并进行处理
                input_path = os.path.join(input_dir, filename)
                img = cv2.imread(input_path)
                img = self.apply_transforms(img,operation)

                # 将处理后的图片保存到输出目录
                output_path = os.path.join(output_dir, operation+prefix+'_'+filename)
                cv2.imwrite(output_path, img)