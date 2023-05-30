import cv2
import numpy as np
from matplotlib import pyplot as plt
import Params as P

def ROImouseCallback(event, x, y, flags, param):
    """
    ROI鼠标回调函数，用于获取点击位置的像素值

    Args:
        event: 鼠标事件
        x: 鼠标点击的x坐标
        y: 鼠标点击的y坐标
        flags: 鼠标事件的标志位
        param: 额外传递的参数（图像）

    Returns:
        None
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        param = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        print(param[y, x])  # 输出点击位置的像素值（HSV颜色空间）

def drawROI(frameLeft, GP: P.GlobalPara, color=(0, 255, 0)):
    """
    在左图像上绘制ROI矩形框

    Args:
        frameLeft: 左图像帧
        GP: 全局参数对象
        color: 矩形框颜色

    Returns:
        None
    """

    imgL_rectified = cv2.remap(frameLeft, P.mapLx, P.mapLy, cv2.INTER_LINEAR)  # 左图像校正

    cv2.namedWindow('ROI', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('ROI', 800, 600)
    cv2.setMouseCallback('ROI', ROImouseCallback, imgL_rectified)  # 设置鼠标回调函数

    cv2.rectangle(imgL_rectified, GP.ROILeftP1, GP.ROILeftP2, color, 2)  # 绘制矩形框

    cv2.imshow('ROI', imgL_rectified)

def drawPoints(frameLeft, points, color=(0, 0, 255)):
    """
    在左图像上绘制点集

    Args:
        frameLeft: 左图像帧
        points: 待绘制的点集
        color: 点的颜色

    Returns:
        None
    """

    points = np.array(points, dtype=np.int32)
    for p in points:
        cv2.circle(frameLeft, [p[1], p[0]], 2, color, 2)  # 绘制点

class visual:
    def __init__(self, xrange, yrange, zrange):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange

        # 调整子图之间的间距
        self.fig.tight_layout()

    def update(self, target, ThreeD):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.set_title('X-Y')
        self.ax2.set_title('Y-Z')

        self.ax1.set_aspect('equal')
        self.ax2.set_aspect('equal')

        self.ax1.set_xlim(self.xrange[0], self.xrange[1])
        self.ax1.set_ylim(self.yrange[0], self.yrange[1])

        self.ax2.set_xlim(self.yrange[0], self.yrange[1])
        self.ax2.set_ylim(self.zrange[0], self.zrange[1])

        target = np.array(target, dtype=np.int32)

        points = ThreeD[target[:, 0], target[:, 1], :]
        points = points[points[:, 2] < 500]
        points = points[points[:, 2] > 200]
        points = np.array(points, dtype=np.float32)
        min = np.argmin(points[:, 1])
        points = points - points[min]
        points[:, 2] = points[:, 2] * (-1)

        # 将三维点集拆分为x、y、z坐标数组
        x = points[:, 0] * (-1)
        y = points[:, 1]
        z = points[:, 2]

        self.ax1.scatter(x, y, c=z, cmap='viridis')  # 在X-Y平面上绘制散点图
        self.ax2.scatter(y, z, c=x, cmap='viridis')  # 在Y-Z平面上绘制散点图

        plt.draw()
        plt.pause(0.01)
