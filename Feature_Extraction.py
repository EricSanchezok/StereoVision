import cv2
import numpy as np
import Params as P
import Convolution as COV


def HSV_segmentation(RGBimage, lower, upper, mask) -> object:
    """
    对RGB图像进行HSV颜色空间分割

    Args:
        RGBimage: 输入的RGB图像
        lower: HSV的下界
        upper: HSV的上界
        mask: 分割掩码

    Returns:
        output: 分割结果图像
    """
    input = RGBimage.copy()

    # 使用掩码进行图像位与操作
    input = cv2.bitwise_and(input, input, mask=mask)

    # 将图像转换为HSV颜色空间
    HSVimage = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)

    # 根据HSV下界和上界进行分割
    output = cv2.inRange(HSVimage, lower, upper)

    # 对分割结果进行膨胀操作
    kernel = np.ones((7, 7), np.uint8)
    output = cv2.dilate(output, kernel, iterations=1)

    # 对分割结果进行腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    output = cv2.erode(output, kernel, iterations=2)

    return output


def Centerline_Extraction(SEGimage) -> object:
    """
    提取中心线

    Args:
        SEGimage: 分割图像

    Returns:
        skeleton: 中心线图像
        target: 中心线上的目标点坐标
    """
    input = SEGimage.copy()

    finished = False
    size = np.size(input)
    skeleton = np.zeros(input.shape, np.uint8)
    initialsize = cv2.countNonZero(input)
    zeros = initialsize

    while (not finished):
        kersize = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kersize, kersize))

        eroded = cv2.erode(input, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(input, temp)

        skeleton = cv2.bitwise_or(skeleton, temp)
        input = eroded.copy()

        zeros = size - cv2.countNonZero(input)
        if zeros == size:
            finished = True

    # 对中心线进行膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    skeleton = cv2.dilate(skeleton, kernel, iterations=2)

    # 对中心线进行腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    skeleton = cv2.erode(skeleton, kernel, iterations=2)

    # 获取中心线上的目标点坐标
    target = np.where(skeleton == 255)
    target = np.array(target).T

    return skeleton, target


def getTargetPoints(frameLeft, GP: P.GlobalPara, debug=False):
    """
    获取目标点集

    Args:
        frameLeft: 左摄像头图像帧
        GP: 全局参数
        debug: 是否开启调试模式
        imgOriginal: 原始图像（用于调试）

    Returns:
        covP: 目标点集
    """
    imgL_rectified = cv2.remap(frameLeft, P.mapLx, P.mapLy, cv2.INTER_LINEAR)

    maskLeft = np.zeros(imgL_rectified.shape[:2], dtype=np.uint8)
    maskLeft[GP.ROILeftP1[1]:GP.ROILeftP2[1], GP.ROILeftP1[0]:GP.ROILeftP2[0]] = 255

    SEGimage = HSV_segmentation(imgL_rectified, GP.lower, GP.upper, mask=maskLeft)

    SKEimage, TargetPoints = Centerline_Extraction(SEGimage)

    start = np.argmin(TargetPoints[:, 0])
    start = TargetPoints[start]

    covP = COV.convolution(SEGimage, start=start, size=[20, 20], step=10, max_pixels=30, debug=debug)

    if debug:
        combinedOut = np.hstack([SEGimage, SKEimage])
        cv2.namedWindow('SEG&SKEcombinedOut', cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow('SEG&SKEcombinedOut', 1600, 600)
        cv2.imshow('SEG&SKEcombinedOut', combinedOut)

    return covP
