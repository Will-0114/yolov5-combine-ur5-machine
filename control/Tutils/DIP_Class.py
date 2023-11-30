## This program is to extract desired color by HSV
## using cv2 trackbar to find a feasibe color lower, higher bound
##  
import os
import cv2
import numpy as np
import time
from datetime import datetime
from glob import glob

class DIP(object):
    stop = False
    scale = 0.6
    hsv_filename = "./config/hsv.txt"
    lower_hsv=[0, 0, 0]
    higher_hsv = [179, 255, 255]
    defect_name = "./SB_Defect_Bank/SB"
    def __init__(self):
        self.ilowH = 0
        self.ihighH = 179
        self.ilowS = 0
        self.ihighS = 255
        self.ilowV = 0
        self.ihighV = 255
        return

    def get_TrackBar(self, x):  ## callback function
        # get trackbar positions
        self.ilowH = cv2.getTrackbarPos('lowH', 'HSV_Control')
        self.ihighH = cv2.getTrackbarPos('highH', 'HSV_Control')
        self.ilowS = cv2.getTrackbarPos('lowS', 'HSV_Control')
        self.ihighS = cv2.getTrackbarPos('highS', 'HSV_Control')
        self.ilowV = cv2.getTrackbarPos('lowV', 'HSV_Control')
        self.ihighV = cv2.getTrackbarPos('highV', 'HSV_Control')
        self.lower_hsv = np.array([self.ilowH, self.ilowS, self.ilowV])
        self.higher_hsv = np.array([self.ihighH, self.ihighS, self.ihighV])
        #print(self.lower_hsv, self.higher_hsv)

        return self.lower_hsv, self.higher_hsv

    def video_track(self):
        print("Start web camera...")
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE) ##cv2.WINDOW_NORMAL) ##
        # create trackbars for color change
        cv2.createTrackbar('lowH', 'Image', self.ilowH, 179, self.get_TrackBar) ## Hue:0-179
        cv2.createTrackbar('highH', 'Image', self.ihighH, 179, self.get_TrackBar)
        cv2.createTrackbar('lowS','Image', self.ilowS, 255, self.get_TrackBar) # Saturation: 0-255
        cv2.createTrackbar('highS','Image', self.ihighS, 255, self.get_TrackBar)
        cv2.createTrackbar('lowV','Image', self.ilowV, 255, self.get_TrackBar) # Intensity: 0-255
        cv2.createTrackbar('highV','Image', self.ihighV, 255, self.get_TrackBar)

        while not self.stop:
            # grab the frame
            ret, frame = cap.read()
            if ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                self.lower_hsv, self.higher_hsv = self.get_TrackBar(x=0)
                mask = cv2.inRange(hsv, self.lower_hsv, self.higher_hsv)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
                # show thresholded image
                cv2.imshow('HSV Segmentation', frame)
                
                key = cv2.waitKey(10) & 0xFF # large wait time to remove freezing
                if key == 113 or key == 27:  ## ESC to escape
                    self.stop_video()
                    break
                if key == ord('s'):
                    cv2.imwrite("seg_img.bmp", frame)

        #print(self.lower_hsv, self.higher_hsv)
        return #self.lower_hsv, self.higher_hsv

    def hsv_segment(self, cvImage, low, high): #, threshold = 10):
        frame = cvImage.copy()
        self.lower_hsv = np.array(low)
        self.higher_hsv = np.array(high)
        #print(self.lower_hsv)
        #print(self.higher_hsv)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
##        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR)
        mask = cv2.inRange(hsv, self.lower_hsv, self.higher_hsv)
        image = cv2.bitwise_and(frame, frame, mask=mask)
        #ret, self.thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        return image #self.thresh
    
    def hsv_dynamic_segment(self, cvImage):
        print("Load image...")
        
        cv2.namedWindow('HSV_Control', cv2.WINDOW_NORMAL) ## cv2.WINDOW_AUTOSIZE) ##
        cv2.resizeWindow('HSV_Control', 500,320)  ## control the window size
        # create trackbars for color change
        cv2.createTrackbar('lowH', 'HSV_Control', self.ilowH, 179, self.get_TrackBar) ## Hue:0-179
        cv2.createTrackbar('highH', 'HSV_Control', self.ihighH, 179, self.get_TrackBar)
        cv2.createTrackbar('lowS','HSV_Control', self.ilowS, 255, self.get_TrackBar) # Saturation: 0-255
        cv2.createTrackbar('highS','HSV_Control', self.ihighS, 255, self.get_TrackBar)
        cv2.createTrackbar('lowV','HSV_Control', self.ilowV, 255, self.get_TrackBar) # Intensity: 0-255
        cv2.createTrackbar('highV','HSV_Control', self.ihighV, 255, self.get_TrackBar)
        threshold = 10
        self.stop = False
        while not self.stop:
            # grab the frame
            frame = cvImage.copy()
            if frame is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                self.lower_hsv, self.higher_hsv = self.get_TrackBar(x=0)
                mask = cv2.inRange(hsv, self.lower_hsv, self.higher_hsv)
                self.seg_image = cv2.bitwise_and(frame, frame, mask=mask)
                # show thresholded image
                #self.seg_image = cv2.resize(frame, None, fx = self.scale, fy = self.scale)
                #frame = self.write_text(frame, 10, 20, "ESC to Stop", size = 40)
                #ret, self.thresh = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)
                cv2.imshow('ESC to Stop', self.seg_image) #self.thresh)
                
                key = cv2.waitKey(10) & 0xFF # large wait time to remove freezing
                if key == 113 or key == 27:  ## ESC to escape
                    self.stop_video()
                    break
                if key == ord('s'):
                    now = datetime.now()
                    time_tag = now.strftime("%Y_%m_%d_%H_%M_%S")
                    fn = defect_name+"_"+time_tag+".bmp"
                    cv2.imwrite(fn, self.seg_image)

        #print(self.lower_hsv, self.higher_hsv)
        return self.seg_image #self.lower_hsv, self.higher_hsv
    
    def write_text(self, img, x, y, text, size = 30, color =(0, 255, 0, 0), font = cv2.FONT_HERSHEY_PLAIN, width =1):
        ## cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
        ##cv2.putText(img, text, (x, y), font, size, color, width, cv2.LINE_AA)
        ## cv cannot show non ASCII text, use pil to do that
        #color = (0,255,0,0) #(b,g,r,a) =   ## color setting
        from PIL import ImageFont, ImageDraw, Image
        fontpath = "./simsun.ttc" # <== 这里是宋体路径 
        font = ImageFont.truetype(fontpath, size)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x, y),  text, fill = color, font = font )  # font = font (b, g, r, a)
        img = np.array(img_pil)
        return img
    
    def stop_video(self):
        self.stop = True
        cv2.cv2.destroyAllWindows()
        return

    def read_hsv_param(self):
        with open(self.hsv_filename, 'r') as f:
            lines = f.readlines()
            print(lines)
            line0 = lines[0].split(',')
            print(line0)
            self.lower_hsv = [int(i) for i in line0]
            line1 = lines[1].split(',')
            print(line1)
            self.higher_hsv = [int(i) for i in line1]
            print("HSV params: ", self.lower_hsv, self.higher_hsv)
        return
    
    def connect_blob(self, image, kernel_w, kernel_h):
        image_copy = image.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_h, kernel_w))
        img_closing = cv2.morphologyEx(image_copy, cv2.MORPH_OPEN, kernel)    
        #img_dilate = cv2.(image_copy, kernel)
        return img_closing

    ## Screening out those big blobs, and unlike nucleus blob
    ## input: original image and blob image
    ## return: marked orignal image, marked blob image, and keypoints 
    def find_blob(self, img, isBlack= True, min_threshod = 10, max_threshold = 200,
                  min_area =1500, max_area =2000000000, min_inertia = 0.01,
                  color =(0, 0, 255)): ## input the original image and blob image
        t_start = time.clock()
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        ## Change the color type: 255 as objects
        params.filterByColor = True
        if isBlack:
            params.blobColor = 0  ## black 0, white: 255
        else:
            params.blobColor = 255
            
        # Change thresholds
        params.filterByArea = True
        params.minThreshold = 10;
        params.maxThreshold = 200;
         
        # Filter by Area.  ## 決定blob的面積大小
        params.filterByArea = True
        params.minArea = min_area ##150
        params.maxArea = max_area ##2500

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.1
         
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87
         
        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = min_inertia ##0.1  ## 過濾到太長的形狀
         
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else :
            print("CV2 version: ", ver)
            detector = cv2.SimpleBlobDetector_create(params)

            
        # Detect blobs.
        keypoints = detector.detect(img)
        ## print out the key points information
        #for i, key in enumerate(keypoints):
        #    print(keypoints[i].pt, keypoints[i].size, keypoints[i].angle, keypoints[i].octave)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        #color = (0, 0, 255)
        original_im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #blob_im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        t_end = time.clock()
        print("spent time: ", t_end - t_start, " sec.")
        return original_im_with_keypoints,  keypoints

    def find_contour(self, img, isBlack= True, min_area = 1000):
##        print("Min Area:", min_area)
        t_start = time.clock()
        img_copy = img.copy()
##        print(img_copy.shape)
        if len(img_copy.shape) == 3:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        ## must do a threshold
        if isBlack:
            ret, img_copy = cv2.threshold(img_copy, 5, 255,cv2.THRESH_BINARY_INV)  ## white is object
        else:
            ret, img_copy = cv2.threshold(img_copy, 5, 255,cv2.THRESH_BINARY)  ## white is object
##        print(img_copy.shape)
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 4 :
            _, contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
##            print("Use cv2: 4 version")
            contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##        print("No of contours:", len(contours), type(contours))
        cnt_list = list()
        bbox_list = list()
        for i, cnt in enumerate(contours):
            cnt=contours[i]
            #print(cnt)
            x,y,w,h=cv2.boundingRect(cnt)
            area_rect = w * h
            if area_rect > min_area:
##                print(i)
                img=cv2.rectangle(img,(x, y),(x+w,y+h), (0, 0, 255), 2)
                cnt_list.append(cnt)
                bbox_list.append((x, y, w, h))
            cv2.drawContours(img, contours, -1, (0,255,0), 1)
        t_end = time.clock() 
        print("spent time: ", t_end - t_start, " sec.")
        #print(contours, hierarchy)
        return img, cnt_list, bbox_list
    
    def crop_Image(self, Img, x_start, y_start, x_end, y_end, isShow = False):
        self.cvImg = Img
        self.cropImg = self.cvImg[y_start: y_end, x_start: x_end]
        height, width, channel = self.cropImg.shape
        #print(height)
        heading = "crop image"
        if isShow:
            #cv2.resizeWindow(heading, int(self.scale* self.width), int(self.scale* self.height))
            #cv2.namedWindow(heading, 0);
            cv2.imshow(heading, self.cropImg)
        return self.cropImg
    
    def patch(self, sourceImg, patchImg, x, y):
        """
        This function patch a defect image into a good image (original idea)
        input: source image is a good image with no defect
        input: patchImg is a defective image (color image 3 channel)  改為中心
        input: x, y are the anchor to patch (color image 3 channel)
        return: output the patched image
        """
        (rows, cols) = patchImg.shape[:2]
        ##print("patch image:", rows, cols)
        ## easy way
        black = np.array([0, 0, 0])
        x = int(x - cols/2)
        y = int(y - rows/2)
        for row in range(rows):
            for col in range(cols):
                pixel = patchImg[row, col]
                #print(type(pixel))
                if np.array_equal(pixel, black):
                    continue
                #print(patchImg[row,col])
                ## 修改: x, y 為中心點
                sourceImg[row+y, col+x] = pixel
        ## 考慮加入模糊化
        return sourceImg
    
    ## Basic CV operation
    def flip(self, cvImage, flipCode):  #code :0 vertical, 1: horizontal, -1: both
        cvImg = cv2.flip(cvImage, flipCode)
        return cvImg
    
    def blur(self, cvImage, mask= (3, 3)):
        cvImg = cv2.blur(cvImage, mask)
        return cvImg

    def sharpen(self, cvImage):  ## laplacian filter
        kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
        sharpened = cv2.filter2D(cvImage, -1, kernel) # applying the sharpening kernel to the input image & displaying it
        return sharpened

    def change_brightness(self, cvImage, value): ## 每次修改intensity的數量
        hsv = cv2.cvtColor(cvImage, cv2.COLOR_BGR2HSV) #convert it to hsv
        #hsv[:,:,2] += value ## + ==> bright   - ==> dark
        # for x in range(0, len(hsv)):
        #     for y in range(0, len(hsv[0])):
        #         #print(hsv[x, y][2])
        #         if hsv[x, y][2] + value > 255:
        #             hsv[x, y][2] = 255
        #             #continue
        #         elif hsv[x, y][2] + value < 0:
        #             hsv[x, y][2] = 0
        #         else:
        #             hsv[x, y][2] += value
        # img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        h, s, v = cv2.split(hsv)
        #print(v.shape)
        rows, cols = v.shape
        #v_list = list(v)
        for i in range(rows):
            for j in range(cols):
                if v.item(i, j) == 255:  ## now background is 255
                        continue  ## background
                if value >=0:   
                    lim = 255 - value
                    if v.item(i, j) > lim:
                        v.itemset((i, j), 255)
                    else:
                        v.itemset((i, j),  v.item(i, j) + value)
                else:
                    #value = int(-value)
                    lim = 0 + (-value)
                    if v.item(i, j) < lim:
                        v.itemset((i, j), 0)
                    else:
                        v.itemset((i, j),  v.item(i, j) + value)
        ### 以下是快速簡易寫法，但backgroud 會改變 ###
        # if value >= 0:
        #     lim = 255 - value
        #     v[v > lim] = 255
        #     v[v <= lim] += value
        # else:
        #     value = int(-value)
        #     lim = 0 + value
        #     v[v < lim] = 0
        #     v[v >= lim] -= value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def findAllImages(self, path = "./train/"):
        fileList =[]
        pattern = "*.bmp"
        for path, subdirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    fileList.append(os.path.join(path, name))
                    print( os.path.join(path, name))
        pattern = "*.jpg"
        for path, subdirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    fileList.append(os.path.join(path, name))
                    print( os.path.join(path, name))
        pattern = "*.png"
        for path, subdirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    fileList.append(os.path.join(path, name))
                    print( os.path.join(path, name))
        return fileList

    ## 20180517  新增函數: 搜尋資料夾下所有影像檔
    def findAllImagFiles(self, path = "./train/"):  ## 搜尋目錄下所有相關影像之檔名
        pattern = os.path.join(path, '*.bmp') 
        bmp_files = sorted(glob(pattern))
        #print(type(bmp_files))
        pattern = os.path.join(path, '*.jpg')
        jpg_files = sorted(glob(pattern))
        pattern = os.path.join(path, '*.png')
        png_files = sorted(glob(pattern))
        file_list = bmp_files + jpg_files + png_files
        return file_list  ## 回傳檔名的 list

    def copy_change_intensity(self, path_src= "./images", path_dest ="./images_dark", value = -13, suffix = "d"):
        ## 產生不同亮度之鋼珠##
        #dip = DIP_Class.DIP()
        #suffix = "d"
        #path_src = "./images"
        #path_dest = "./images_dark"
        if not os.path.exists(path_dest):
            os.mkdir(path_dest)
        ## copy all image to dest
        file_list = self.findAllImagFiles(path_src)
        for file in file_list:
            pathname, filename, file_extension = self.split_filename_extension(file)
            print(pathname, filename, file_extension)
            new_fn = path_dest+ "/" + filename + "_" + suffix + file_extension
            cvImage = cv2.imread(file, -1)
            cvImage_d = self.change_brightness(cvImage, value)
            cv2.imwrite(new_fn, cvImage_d)
        return
    
    def split_filename_extension(self, full_filename):
        path_name = os.path.dirname(full_filename)
        filename_w_ext = os.path.basename(full_filename)
        filename, file_extension = os.path.splitext(filename_w_ext)
        return path_name, filename, file_extension
    ## adding face tracking ###
    
def dip_pi():
    dip = DIP()
    dip.read_hsv_param()
    cvImage = cv2.imread("./Station03.jpeg")
    dip.hsv_segment(cvImage, dip.lower_hsv, dip.higher_hsv)
    print("HSV bounds:", dip.lower_hsv, dip.higher_hsv)
    img_close = dip.connect_blob(dip.thresh, kernel_w= 100, kernel_h = 70)
    cv2.imwrite("close.bmp", img_close)
    #cv2.imshow("HSV segment", img_close)
    img_bbox, cnt_list, bbox_list = dip.find_contour(img_close, min_area = 200000)
    print(bbox_list)
    #img_rect = hsv_seg.find_contour(img_close)
    cv2.imshow("BBox", img_bbox)
    cv2.waitKey(-1)
    return

def patch():
    dip = DIP()
    dip.scale = 2
    cvGoodImage = cv2.imread("./Good_Images/001.bmp")
    cvGoodImage = cv2.resize(cvGoodImage, None, fx = 2, fy = 2)
    cv2.imshow("Good image", cvGoodImage)
    cvDefectImage = cv2.imread("./SB_Defect_Bank/SB_Defect2020_02_27_01_33_16.bmp")
    cvDefectImage = cv2.resize(cvDefectImage, None, fx = 0.5, fy = 0.5)
    cv2.imshow("Defect image", cvDefectImage)
    cv2.imwrite("d1_resized.bmp", cvDefectImage)
    dip.hsv_segment(cvDefectImage, (0,0,0), (100,255,158))
    cvNew = dip.patch(cvGoodImage, cvDefectImage, x=150, y=150 )
    cv2.imshow("New image", cvNew)
    #dip.hsv_segment(cvImage, dip.lower_hsv, dip.higher_hsv)
    #dip.hsv_dynamic_segment(cvImage)
    cv2.waitKey(0)
    return

def main():
    dip = DIP()
    # dip.scale = 2
    # cvImg = cv2.imread("./images/001.bmp")
    # cv2.imshow("Good image", cvImg)
    # #flip_img = dip.flip(cvImg, 0)
    # b_img = dip.change_brightness(cvImg, value = -20)
    # cv2.imshow("brighten", b_img)
    # cv2.waitKey(0)
    dip.copy_change_intensity(path_src= "./Dest", path_dest ="./Dest_dark", value = -15, suffix = "d")
    return

if __name__ == "__main__":
    main()
