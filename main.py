import numpy as np
import cv2
import time
import Params as P
import Feature_Extraction as FE
import Stereo_Vision as SV
import Visualation

GP = P.GlobalPara(ROILeftP1=[320, 100], ROILeftP2=[620, 500],
                  lower=np.array([0, 0, 180]), upper=np.array([120, 40, 255]))

capLeft, capRight = cv2.VideoCapture(1, cv2.CAP_DSHOW), cv2.VideoCapture(2, cv2.CAP_DSHOW)
capLeft.set(3, 800), capLeft.set(4, 600), capRight.set(3, 800), capRight.set(4, 600)

SGBM_parameters = P.StereoParamsInit(debug=False)

visual = Visualation.visual([-75, 75], [0, 150], [-100, 50])

if __name__ == '__main__':

    debug = True

    while True:

        time_start = time.time()

        # 读取左右相机的图像帧
        (ret1, frameLeft), (ret2, frameRight) = capLeft.read(), capRight.read()

        time_read = time.time()

        if not ret1 or not ret2:
            # 如果未读取到图像，则退出循环
            print(ret1, ret2, '未读取到图像')
            break

        # 计算视差图和三维点云
        disp, ThreeD = SV.imgsTodisparity(frameLeft, frameRight, SGBM_parameters, debug=debug)

        time_disparity = time.time()

        # 获取目标区域的点集
        target = FE.getTargetPoints(frameLeft, GP, debug=debug)

        time_target = time.time()

        # 在左图像上绘制目标点集和ROI矩形框
        Visualation.drawPoints(frameLeft, target)
        Visualation.drawROI(frameLeft, GP)

        # 更新三维点云的可视化
        visual.update(target, ThreeD)

        time_visual = time.time()

        key = cv2.waitKey(10)

        if key == ord('q'):
            # 如果按下键盘上的'q'键，则关闭窗口并退出循环
            cv2.destroyAllWindows()
            break

        time_end = time.time()
        if debug:
            # 计算每个步骤的运行时间，并打印出来
            print('timeDisparity:', round(time_disparity - time_read, 2), 's',
                'timeTarget:', round(time_target - time_disparity, 2), 's',
                'timeVisual:', round(time_visual - time_target, 2), 's',
                'timeCost:', round(time_end - time_start, 2), 's',
                'FPS:', round(1 / (time_end - time_start), 2), end='\r')
