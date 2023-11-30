import numpy as np
import cv2
from Tutils.yolov5_utils import yolov5_setup, yolov5_load_model, yolov5_predict

def angle(file, x1, y1, x2, y2):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img = img[y1: y2, x1: x2]
    cv2.imshow("img", img)
    cv2.waitKey(-1)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #print(hsv_img.shape)
    #print('hsv', hsv_img)
    lower = np.array([0, 55, 60]) 
    upper = np.array([179, 255, 193])

    mask = cv2.inRange(hsv_img, lower, upper)
    mask = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    binaryIMG = cv2.Canny(blurred, 50, 120, apertureSize = 3)
    contours, hierarchy = cv2.findContours(binaryIMG, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(1)

    for i in range(len(contours)-1):
        cnt = contours[i]
        cnt1 = contours[i+1]
        #print('cnt=',cnt)
        rect = cv2.minAreaRect(cnt)
        rect1 = cv2.minAreaRect(cnt1)
        #center = rect[0]
        #print('rect=', rect, 'rect1=', rect1)
        if round(rect[0][0]) == round(rect1[0][0]):
            continue
        else:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print('box', box)
            print("angle = ", rect[2])
            
            cX = round(rect[0][0])
            cY = round(rect[0][1])
            print('cX=', cX, 'cY=', cY)
            cv2.circle(img, (cX, cY), 5, (1, 227, 254), -1)
            return rect[2],cX,cY
            #for n in range(len(yolov5_predict.det_list)):
            #    lst = yolov5_predict.det_list[n]
            #    if (cX, cY)  >= (lst[0], lst[1]) and (cX, cY)  <= (lst[2], lst[3]):
            #        cv2.drawContours(img, [box], -1,(0, 0, 255), 2)
            #        cv2.imshow("img", img)
            #        cv2.waitKey(-1)
    

if __name__ == "__main__":
    x1 = 180
    y1 = 100
    x2 = 180
    y2 = 400
    file = "Save_Images\\color_Img_2021_01_11_00_04_02.jpg"
    angle = angle(file, x1, y1, x2, y2)
    print(angle)
