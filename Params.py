import cv2
import numpy as np

# 相机内参矩阵
K1 = np.array([[603.203230962510, 0, 0],
               [0, 603.282035569803, 0],
               [400.442038630029, 301.405795942410, 1]], dtype=float).T

D1 = np.array([0.0846, -0.1510, 0, 0, 0], dtype=float)

K2 = np.array([[605.360594316701, 0, 0],
               [0, 604.975943329530, 0],
               [400.089763847830, 299.044031919448, 1]], dtype=float).T

D2 = np.array([0.0886, -0.1568, 0, 0, 0], dtype=float)

R = np.array([[0.999034572196626, 0.0257176721743424, -0.0356163571107109],
              [-0.0258012805637891, 0.999665305337157, -0.00188976888444957],
              [0.0355558360494205, 0.00280689207150011, 0.999363749532534]], dtype=float).T

T = np.array([[-77.7771596068266], [-1.01312800155798], [2.62572217808557]], dtype=float)

F = np.array([[-8.97965080103296e-08, -7.17241155026077e-06, 0.000498161266153330],
              [-5.33652440999904e-07, -1.32009826881906e-06, 0.129244355632227],
              [-0.00153924655477953, -0.125656116896911, 0.253033715598527]], dtype=float)

E = np.array([[-0.0327896857626352, -2.61938744612640, -1.02887295018107],
              [-0.194742286493101, -0.481797741783444, 77.8197334860518],
              [-1.04641211072644, -77.7754173526746, -0.449957959277595]], dtype=float)

RNew = cv2.Rodrigues(R)[0]  # 旋转矩阵
size = (800, 600)  # 图像尺寸

# 立体校正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)

# 计算畸变校正的映射关系
mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)


class GlobalPara:
    def __init__(self, ROILeftP1, ROILeftP2, lower, upper) -> None:
        """
        全局参数类，用于存储一些参数

        Args:
            ROILeftP1: 左图像感兴趣区域的左上角坐标
            ROILeftP2: 左图像感兴趣区域的右下角坐标
            lower: HSV 颜色空间的下界
            upper: HSV 颜色空间的上界
        """
        ROIxleftP1, ROIyleftP1 = int(ROILeftP1[0]), int(ROILeftP1[1])  # 左上角坐标
        ROIxleftP2, ROIyleftP2 = int(ROILeftP2[0]), int(ROILeftP2[1])  # 右下角坐标

        self.ROILeftP1 = [ROIxleftP1, ROIyleftP1]
        self.ROILeftP2 = [ROIxleftP2, ROIyleftP2]

        self.lower = lower  # HSV 颜色空间的下界
        self.upper = upper  # HSV 颜色空间的上界


class StereoParamsInit:
    def __init__(self, debug=False):
        self.debug = debug
        self.window_name = "bar"

        if self.debug:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow(self.window_name, 800, 600)

            # 创建调节参数的滑动条
            cv2.createTrackbar("minDisparity", self.window_name, 0, 100,
                               lambda x: cv2.setTrackbarPos("minDisparity", self.window_name, max(x, 0)))
            cv2.createTrackbar("num", self.window_name, 1, 20,
                               lambda x: cv2.setTrackbarPos("num", self.window_name, max(x, 1)))
            cv2.createTrackbar("blockSize", self.window_name, 5, 50,
                               lambda x: cv2.setTrackbarPos("blockSize", self.window_name,
                                                            max(x + 1 if x % 2 == 0 else x, 5)))
            cv2.createTrackbar("window_size", self.window_name, 1, 20,
                               lambda x: cv2.setTrackbarPos("window_size", self.window_name,
                                                            max(x + 1 if x % 2 == 0 else x, 1)))
            cv2.createTrackbar("disp12MaxDiff", self.window_name, 1, 100,
                               lambda x: cv2.setTrackbarPos("disp12MaxDiff", self.window_name, max(x, 1)))
            cv2.createTrackbar("uniquenessRatio", self.window_name, 1, 10,
                               lambda x: cv2.setTrackbarPos("uniquenessRatio", self.window_name, max(x, 1)))
            cv2.createTrackbar("speckleWindowSize", self.window_name, 1, 200,
                               lambda x: cv2.setTrackbarPos("speckleWindowSize", self.window_name, max(x, 1)))
            cv2.createTrackbar("speckleRange", self.window_name, 1, 200,
                               lambda x: cv2.setTrackbarPos("speckleRange", self.window_name, max(x, 1)))
            cv2.createTrackbar("preFilterCap", self.window_name, 1, 200,
                               lambda x: cv2.setTrackbarPos("preFilterCap", self.window_name, max(x, 1)))

    def getParams(self):
        """
        获取立体匹配算法的参数
        """
        SGBM_parameters = {
            'minDisparity': cv2.getTrackbarPos("minDisparity", self.window_name) if self.debug else 50,
            'numDisparities': 16 * (cv2.getTrackbarPos("num", self.window_name) if self.debug else 16),
            'blockSize': cv2.getTrackbarPos("blockSize", self.window_name) if self.debug else 5,

            'P1': 8 * 3 * (cv2.getTrackbarPos("window_size", self.window_name) if self.debug else 3),
            'P2': 32 * 3 * (cv2.getTrackbarPos("window_size", self.window_name) if self.debug else 3),

            'disp12MaxDiff': cv2.getTrackbarPos("disp12MaxDiff", self.window_name) if self.debug else 50,
            'uniquenessRatio': cv2.getTrackbarPos("uniquenessRatio", self.window_name) if self.debug else 1,
            'speckleWindowSize': cv2.getTrackbarPos("speckleWindowSize", self.window_name) if self.debug else 400,
            'speckleRange': cv2.getTrackbarPos("speckleRange", self.window_name) if self.debug else 400,
            'preFilterCap': cv2.getTrackbarPos("preFilterCap", self.window_name) if self.debug else 200,
            'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
        }

        return SGBM_parameters
