import numpy as np
import cv2
import math

class vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return vector(int(self.x + other.x), int(self.y + other.y))
    
    def __sub__(self, other):
        return vector(int(self.x - other.x), int(self.y - other.y))
    
    def __mul__(self, other):
        return vector(self.x * other, self.y * other)
    
    def __str__(self):
        return "vector: x = %d, y = %d" % (self.x, self.y)
    
    def length(self):
        return int(math.sqrt(self.x**2 + self.y**2))


def convolution(image_conv, start, size, step, max_pixels, debug=False, imgOriginal=None):
    """
    卷积函数，用于提取目标点集

    Args:
        image_conv: 输入图像
        start: 起始点坐标
        size: 区域大小
        step: 步长
        max_pixels: 最大像素数
        debug: 是否开启调试模式
        imgOriginal: 原始图像（用于调试）

    Returns:
        result: 目标点集
    """
    result = []

    startPoint = vector(start[1], start[0])
    lastOriMeanVec = startPoint

    region = image_conv[int(startPoint.y-size[0]/2):int(startPoint.y+size[0]/2), int(startPoint.x-size[1]/2):int(startPoint.x+size[1]/2)]

    if debug:
        cv2.namedWindow("region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("region", 100, 100)

    while(np.array(np.where(region != 0)).shape[1] > max_pixels):
        region_mean = np.array(np.where(region != 0))
        if region_mean.shape[1] == 0:
            break
        region_mean = region_mean.mean(axis=1).astype(int)
        region_meanVec = vector(region_mean[1], region_mean[0])
        ori_meanVec = vector(startPoint.x-size[1]/2, startPoint.y-size[0]/2) + region_meanVec
        point = [ori_meanVec.y, ori_meanVec.x]
        result.append(point)

        if debug:
            cv2.imshow("region", region)

        forwardVec = ori_meanVec - lastOriMeanVec
        lastOriMeanVec = ori_meanVec
        forwardVec = vector((forwardVec.x*step / forwardVec.length()),int(forwardVec.y*step / forwardVec.length()))

        startPoint = ori_meanVec + forwardVec

        region = image_conv[int(startPoint.y-size[0]/2):int(startPoint.y+size[0]/2), int(startPoint.x-size[1]/2):int(startPoint.x+size[1]/2)]

    return result
