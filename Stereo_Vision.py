import cv2
import Params as P
import numpy as np

def get_point_cloud(disparity, Q):
    """
    根据视差图和投影矩阵计算深度图

    Args:
        disparity: 视差图
        Q: 投影矩阵

    Returns:
        Points3D: 三维点云
    """

    Points3D = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)  # 根据视差图和投影矩阵计算三维点云

    return Points3D


def imgsTodisparity(frameLeft, frameRight, SGBM_parameters: P.StereoParamsInit, debug=False):
    """
    将左右图像转换为视差图和三维点云

    Args:
        frameLeft: 左图像帧
        frameRight: 右图像帧
        SGBM_parameters: 立体匹配算法参数
        debug: 是否启用调试模式

    Returns:
        disp: 视差图
        ThreeD: 三维点云
    """

    imgL_rectified = cv2.remap(frameLeft, P.mapLx, P.mapLy, cv2.INTER_LINEAR)  # 左图像校正

    imgR_rectified = cv2.remap(frameRight, P.mapRx, P.mapRy, cv2.INTER_LINEAR)  # 右图像校正

    imgL_gray = cv2.cvtColor(imgL_rectified, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    imgR_gray = cv2.cvtColor(imgR_rectified, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

    parameters = SGBM_parameters.getParams()  # 获取立体匹配算法参数
    left_matcher = cv2.StereoSGBM_create(**parameters)  # 创建立体匹配对象

    disp = left_matcher.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0  # 计算视差图

    ThreeD = get_point_cloud(disp, P.Q)  # 根据视差图和投影矩阵计算三维点云

    if debug:
        cv2.namedWindow('dispVisualized', cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow('dispVisualized', 800, 600)

        dispVisualized = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow('dispVisualized', dispVisualized)

    return disp, ThreeD
