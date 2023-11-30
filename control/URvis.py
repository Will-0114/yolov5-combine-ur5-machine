from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import logging
import argparse
import imutils
import cv2
import os
import os.path
import matplotlib.pyplot as plt
import time
import urx
# from yoloUse.yoloOpencv import opencvYOLO
# from yoloUse.yoloPydarknet import pydarknetYOLO

x_list = []
y_list = []
#pixelsPerMetric = 8.67
line_x = 0
line_y = 0
refObj = None 
width = 960
height = 720           
# 影像處理


class UR_cap():

        def cap(self):  # 開啟移動偵測鏡頭
                cap = cv2.VideoCapture(1)  # 改變括號內數字可選擇webcam
                # 設定影像尺寸
                width = 960
                height = 720

                # 設定擷取影像尺寸大小
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 修改解析度 寬
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 修改解析度 高
                area = width * height  # 面積

                # 初始化平均影像
                ret, frame = cap.read()
                avg = cv2.blur(frame, (4, 4))
                avg_float = np.float32(avg)

                while (cap.isOpened()):
                        ret, frame = cap.read()
                        if ret == False:
                                break
                        blur = cv2.blur(frame, (4, 4))
                        diff = cv2.absdiff(avg, blur)
                        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(
                            gray, 25, 255, cv2.THRESH_BINARY)

                        # 使用型態轉換函數去除雜訊
                        kernel = np.ones((5, 5), np.uint8)
                        thresh = cv2.morphologyEx(
                            thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                        thresh = cv2.morphologyEx(
                            thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

                        # 產生等高線
                        cnts = cv2.findContours(
                            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts)
                        cv2.drawContours(frame, cnts, -1, (0, 0, 255), 2)

                        for c in cnts:
                        # 忽略太小的區域
                                if cv2.contourArea(c) < 2000:
                                        continue
                                # 偵測到物體，可以自己加上處理的程式碼在這裡...

                                (x, y, w, h) = cv2.boundingRect(c)  # 計算等高線的外框範圍
                                cv2.rectangle(
                                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 畫出外框

                        # 畫出等高線（除錯用）
                        cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)

                        # 顯示偵測結果影像
                        cv2.imshow('frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        elif cv2.waitKey(1) & 0xFF == ord('0'):
                                cv2.imwrite(
                                    'C:/Users/user/Desktop/Detect.jpg', frame)
                                break
                        # 更新平均影像
                        cv2.accumulateWeighted(blur, avg_float, 0.01)
                        avg = cv2.convertScaleAbs(avg_float)
                        image = cv2.imread(r"C:/Users/user/Desktop/Detect.jpg")
                cap.release()
                cv2.destroyAllWindows()
                return image

        def cap00(self):
                cap = cv2.VideoCapture(1)  # 改變括號內數字可選擇webcam
                # 設定影像尺寸
                width = 960
                height = 720

                # 設定擷取影像尺寸大小
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 修改解析度 寬
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 修改解析度 高
                while(True):
                        ret, frame = cap.read()
                        if ret == True:
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                cv2.imshow('frame', frame)
                                cv2.imwrite('C:/Users/user/Desktop/success.jpg', frame)
                                break
                cap.release()
                cv2.destroyAllWindows()
                return 

        def midpoint(self, ptA, ptB):  # 定义一个中点函数，后面会用到
	        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

        def dis1(self, image):
                image = cv2.imread(r"C:/Users/user/Desktop/Detect.jpg")
                # convert the image to grayscale, blur it, and detect edges
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (11, 11), 0)
                edged = cv2.Canny(gray, 35, 125)
                # find the contours in the edged image and keep the largest one
                # we'll assume that this is our piece of paper in the image
                cnts = cv2.findContours(
                    edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                c = max(cnts, key=cv2.contourArea)
                # 求最大面積
                return cv2.minAreaRect(c)
                # cv2.minAreaRect() c代表點集，返回rect[0]是最小外接矩形中心點座標，
                # rect[1][0]是width，rect[1][1]是height，rect[2]是角度
        
        def x_y_center(self):  # 圓形中心偵測
                global line_x
                global line_y
                image = cv2.imread(r"C:/Users/user/Desktop/Detect.jpg")  # 读取输入图片
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 输入图片灰度化
                gray = cv2.GaussianBlur(gray, (11, 11), 0)  # 对灰度图片执行高斯滤波
                edged = cv2.Canny(gray, 50, 100)  # 对滤波结果做边缘检测获取目标
                # 使用膨胀和腐蚀操作进行闭合对象边缘之间的间隙
                edged = cv2.dilate(edged, None, iterations=1)
                edged = cv2.erode(edged, None, iterations=1)
                # 在边缘图像中寻找物体轮廓（即物体）
                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
                i = 0
                img=image.copy()
                (cnts, _)=contours.sort_contours(cnts)
                for c in cnts:
                # CV2.moments會傳回一系列的moments值，我們只要知道中點X, Y的取得方式是如下進行即可。
                        if cv2.contourArea(c) < 500:
                                continue
                        box=cv2.minAreaRect(c)
                        M = cv2.moments(c)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        print("像素點位:", cX, cY)
                        x_list.append(cX)
                        y_list.append(cY)
                        cv2.circle(img, (cX, cY), 3, (0, 0, 255), -1)
                        i +=1
                        cv2.putText(img, "#%d" %(i), (int(cX-20), int(cY)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (252, 197, 5), 3)
                        # (x, y, w, h) = cv2.boundingRect(c)# 計算等高線的外框範圍
                        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("threshold", img)
                if len(y_list) > 0:  # 有讀到物件
                        print("Detected")
                        for (x,y) in zip(x_list,y_list):
                                print(x,y)
                        cv2.imwrite('C:/Users/user/Desktop/choose.jpg', img)
                        return True
                else:
                        print("not detected")
                        return False

        

        def pixel_distance(self,pcx,pcy,num):  # xy距離
                w = 8.6
                w = float(w)
                pcx = width/2
                pcy = height / 2
                print(pcx,pcy)
                image=cv2.imread(r'C:/Users/user/Desktop/Detect.jpg')
                # 输入图片灰度化
                gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # 对灰度图片执行高斯滤波
                gray=cv2.GaussianBlur(gray, (11, 11), 0)
                # 对滤波结果做边缘检测获取目标
                edged=cv2.Canny(gray, 50, 100)
                # 使用膨胀和腐蚀操作进行闭合对象边缘之间的间隙
                edged=cv2.dilate(edged, None, iterations=1)
                edged=cv2.erode(edged, None, iterations=1)

                # 在边缘图像中寻找物体轮廓（即物体）
                cnts=cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
                cnts=imutils.grab_contours(cnts)
                cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
                # 对轮廓按照从左到右进行排序处理
                (cnts, _)=contours.sort_contours(cnts)
                i =0
                pixelsPerMetric = None
                for c in cnts:
                        if cv2.contourArea(c) < 500:
                                continue
                        # 根据物体轮廓计算出外切矩形框
                        box=cv2.minAreaRect(c)
                        i +=1
                        orig=image.copy()
                        M=cv2.moments(c)
                        cX=int(M["m10"] / M["m00"])
                        cY=int(M["m01"] / M["m00"])
                        cv2.circle(orig, (cX, cY), 3, (0, 0, 255), -1)
                        cv2.putText(orig, "#%d" %(i), (int(cX-20), int(cY)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (252, 197, 5), 3)
                        box=cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                        box=np.array(box, dtype="int")
                        # 按照top-left, top-right, bottom-right, bottom-left的顺序对轮廓点进行排序，并绘制外切的BB，用绿色的线来表示
                        box=perspective.order_points(box)
                        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)


                        # 绘制BB的4个顶点，用红色的小圆圈来表示
                        for (x, y) in box:
                                cv2.circle(orig, (int(x), int(y)),
                                           5, (0, 0, 255), -1)

                        # 分别计算top-left 和top-right的中心点和bottom-left 和bottom-right的中心点坐标
                        (tl, tr, br, bl)=box
                        (tltrX, tltrY)=self.midpoint(tl, tr)
                        (blbrX, blbrY)=self.midpoint(bl, br)

                        # 分别计算top-left和top-right的中心点和top-righ和bottom-right的中心点坐标
                        (tlblX, tlblY)=self.midpoint(tl, bl)
                        (trbrX, trbrY)=self.midpoint(tr, br)

                        # 绘制BB的4条边的中心点，用蓝色的小圆圈来表示
                        cv2.circle(orig, (int(tltrX), int(tltrY)),5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(blbrX), int(blbrY)),5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(tlblX), int(tlblY)),5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(trbrX), int(trbrY)),5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(pcx), int(pcy)), 5, (0, 80, 255), -1)

                        # 在中心点之间绘制直线，用紫红色的线来表是
                        cv2.line(orig, (int(tltrX), int(tltrY)),
                                 (int(blbrX), int(blbrY)), (255, 0, 255), 2)
                        cv2.line(orig, (int(tlblX), int(tlblY)),
                                 (int(trbrX), int(trbrY)), (255, 0, 255), 2)
                        cv2.line(orig, (int(pcx), int(pcy)), (int(cX), int(cY)),(18, 80, 255), 2)

                        # 计算两个中心点之间的欧氏距离，即图片距离
                        dA=dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                        dB=dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                        d1 = dist.euclidean((pcx, pcy), (cX, cY))

                        # 初始化测量指标值，参考物体在图片中的宽度已经通过欧氏距离计算得到，参考物体的实际大小已知
                        
                        if pixelsPerMetric is None:
                                pixelsPerMetric = dB / w
                        # 计算目标的实际大小（宽和高）
                        dimA=dA / pixelsPerMetric
                        dimB=dB / pixelsPerMetric
                        dim1 = d1 /pixelsPerMetric

                        # 在图片中绘制结果
                        cv2.putText(orig, "{:.1f}cm".format(dim1),
                                (int(pcx ), int(pcy )
                                 ), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (180, 220, 0), 2)
                        cv2.putText(orig, "{:.1f}cm".format(dimA),
                                (int(tltrX - 15), int(tltrY - 10)
                                 ), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 2)
                        cv2.putText(orig, "{:.1f}cm".format(dimB),
                                (int(trbrX + 10), int(trbrY)
                                 ), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 2)

                        cv2.imshow("Image", orig)
                        cv2.waitKey(0)
                if num ==None:
                        print('未選取物品')
                        x,y = None
                else:
                        x = (x_list[num-1]-pcx)/pixelsPerMetric
                        y = (y_list[num-1]-pcy)/pixelsPerMetric
                        #print(dA/dimA,pixelsPerMetric)
                        print('第',num,"個")
                        print("x,y要移動的距離(CM)", abs(x), abs(y))

                return  x,y

