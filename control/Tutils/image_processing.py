# -*-coding: utf-8 -*-
"""
    @Project: IntelligentManufacture
    @File   : image_processing.py
    @Author : panjq, FC Tien
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-02-14 15:34:50
"""
 
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def show_image(title, image):
    '''
    调用matplotlib显示RGB图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()
 
def cv_show_image(title, image):
    '''
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入RGB图像
    :return:
    '''
    channels=image.shape[-1]
    if channels==3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    cv2.imshow(title,image)
    cv2.waitKey(0)
 
def read_image(filename, resize_height=None, resize_width=None, normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的RGB图片数据
    '''
 
    bgr_image = cv2.imread(filename)
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
 
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    rgb_image = resize_image(rgb_image,resize_height,resize_width)
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    # show_image("src resize image",image)
    return rgb_image
 
def fast_read_image_roi(filename, orig_rect, ImreadModes=cv2.IMREAD_COLOR, normalization=False):
    '''
    快速读取图片的方法
    :param filename: 图片路径
    :param orig_rect:原始图片的感兴趣区域rect
    :param ImreadModes: IMREAD_UNCHANGED
                        IMREAD_GRAYSCALE
                        IMREAD_COLOR
                        IMREAD_ANYDEPTH
                        IMREAD_ANYCOLOR
                        IMREAD_LOAD_GDAL
                        IMREAD_REDUCED_GRAYSCALE_2
                        IMREAD_REDUCED_COLOR_2
                        IMREAD_REDUCED_GRAYSCALE_4
                        IMREAD_REDUCED_COLOR_4
                        IMREAD_REDUCED_GRAYSCALE_8
                        IMREAD_REDUCED_COLOR_8
                        IMREAD_IGNORE_ORIENTATION
    :param normalization: 是否归一化
    :return: 返回感兴趣区域ROI
    '''
    # 当采用IMREAD_REDUCED模式时，对应rect也需要缩放
    scale=1
    if ImreadModes == cv2.IMREAD_REDUCED_COLOR_2 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_2:
        scale=1/2
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_4 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_4:
        scale=1/4
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_8 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_8:
        scale=1/8
    rect = np.array(orig_rect)*scale
    rect = rect.astype(int).tolist()
    bgr_image = cv2.imread(filename,flags=ImreadModes)
 
    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 3:  #
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    else:
        rgb_image=bgr_image #若是灰度图
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    roi_image=get_rect_image(rgb_image , rect)
    # show_image_rect("src resize image",rgb_image,rect)
    # cv_show_image("reROI",roi_image)
    return roi_image
 
def resize_image(image,resize_height, resize_width):
    '''
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape=np.shape(image)
    height=image_shape[0]
    width=image_shape[1]
    if (resize_height is None) and (resize_width is None):#错误写法：resize_height and resize_width is None
        return image
    if resize_height is None:
        resize_height=int(height*resize_width/width)
    elif resize_width is None:
        resize_width=int(width*resize_height/height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image
    
def scale_image(image,scale):
    '''
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    '''
    image = cv2.resize(image,dsize=None, fx=scale[0],fy=scale[1])
    return image
 
 
def get_rect_image(image,rect):
    '''
    :param image:
    :param rect: [x,y,w,h]
    :return:
    '''
    x, y, w, h=rect
    cut_img = image[y:(y+ h),x:(x+w)]
    return cut_img
def scale_rect(orig_rect,orig_shape,dest_shape):
    '''
    对图像进行缩放时，对应的rectangle也要进行缩放
    :param orig_rect: 原始图像的rect=[x,y,w,h]
    :param orig_shape: 原始图像的维度shape=[h,w]
    :param dest_shape: 缩放后图像的维度shape=[h,w]
    :return: 经过缩放后的rectangle
    '''
    new_x=int(orig_rect[0]*dest_shape[1]/orig_shape[1])
    new_y=int(orig_rect[1]*dest_shape[0]/orig_shape[0])
    new_w=int(orig_rect[2]*dest_shape[1]/orig_shape[1])
    new_h=int(orig_rect[3]*dest_shape[0]/orig_shape[0])
    dest_rect=[new_x,new_y,new_w,new_h]
    return dest_rect
 
def show_image_rect(win_name,image,rect):
    '''
    :param win_name:
    :param image:
    :param rect:
    :return:
    '''
    x, y, w, h=rect
    point1=(x,y)
    point2=(x+w,y+h)
    cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    cv_show_image(win_name, image)
 
def rgb_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
 
def save_image(image_path, rgb_image,toUINT8=True):
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image * 255, dtype=np.uint8)
    if len(rgb_image.shape) == 2:  # 若是灰度图则转为三通道
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)
 
def combime_save_image(orig_image, dest_image, out_dir,name,prefix):
    '''
    命名标准：out_dir/name_prefix.jpg
    :param orig_image:
    :param dest_image:
    :param image_path:
    :param out_dir:
    :param prefix:
    :return:
    '''
    dest_path = os.path.join(out_dir, name + "_"+prefix+".jpg")
    save_image(dest_path, dest_image)
 
    dest_image = np.hstack((orig_image, dest_image))
    save_image(os.path.join(out_dir, "{}_src_{}.jpg".format(name,prefix)), dest_image)

## written by Tien 20200515
def find_no_image_in_dir(path):
    import os
    import gc
    print("[MSG]: Reading the data by classes and data balancing ...")
    No_Image_in_dirs = list()
    dirs = os.listdir(path)
    print("Classes: ", dirs)
    count = 0
    class_list = list()
    ## calcuate the image no in each directory        
    for dir in dirs:
        fullpath = os.path.join(path, dir)
        class_list.append(dir)  ## class label
        if os.path.isdir(fullpath):
            ##files_path = os.path.join(fullpath, '*.jpg')  
            ##file = sorted(glob(files_path))
            file = findAllImagFiles(fullpath)
            No_Image_in_dirs.append(len(file))
            print(dir, " = ", No_Image_in_dirs[count])
            count +=1
    print("Max No: ", max(No_Image_in_dirs))
    del file
    gc.collect()
    save_class_list(class_list)
    return No_Image_in_dirs, class_list

def findAllImagFiles(path = "./train/"):  ## 搜尋目錄下所有相關影像之檔名
    from glob import glob
    pattern = os.path.join(path, '*.bmp') 
    bmp_files = sorted(glob(pattern))
    #print(type(bmp_files))
    pattern = os.path.join(path, '*.jpg')
    jpg_files = sorted(glob(pattern))
    pattern = os.path.join(path, '*.jpeg')
    jpeg_files = sorted(glob(pattern))
    pattern = os.path.join(path, '*.png')
    png_files = sorted(glob(pattern))
    file_list = bmp_files + jpg_files + jpeg_files+ png_files
    return file_list  ## 回傳檔名的 list

def save_class_list(class_list):
    with open('./class_list.txt', 'w', encoding='utf8') as f:
        for l in class_list:
            f.write(l + "\n")
    # with open('./class_list.txt', 'w', encoding='utf8') as f:
    #     for l in class_list:
    #         f.write(l + "\n")
    return

def write_train_list(train_list, label_list, filename = "./train.txt"):
    with open(filename, "w") as f:
        for i in range(len(train_list)):
            f.write(str(train_list[i]) +" ," + str(label_list[i]) + "\n")
    return

def create_train_file(image_dir = "./train"):
    No_Image_in_dirs, class_list = find_no_image_in_dir(image_dir)
    import os  
    # ## read all images in training directory
    dirIndex = 0
    dirs = os.listdir(image_dir)
    image_list = list()
    label_list = list()
    for dir in dirs:
        fullpath = os.path.join(image_dir, dir)
        if os.path.isdir(fullpath):
            #files_path = os.path.join(fullpath, '*.jpg')
            #files = sorted(glob(files_path))
            files = findAllImagFiles(fullpath)
            no_of_copy = int(max(No_Image_in_dirs) / No_Image_in_dirs[dirIndex]+0.5)
            print("No of copies: ", no_of_copy)        
            no_of_image = 0
            for f in files:
                for i in range(no_of_copy): ## data balancing
                    #img = load_img(f, target_size=(self.resize_width, self.resize_height, self.no_of_channel)) ## keras read data and reshape
                    #img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), -1) #cvImg = cv2.imread(f, -1) #
                    #img = cv2.resize(img, (self.resize_height, self.resize_width))
                    #new_img = img_to_array(img) 
                    image_list.append(f)
                    label_list.append(dirIndex)      
                    no_of_image +=1
        print(dir, ":", no_of_image)
        dirIndex +=1  ## store 0, 1, 2, 3, 4        
    del files
    n_files = len(image_list)
    print('No of images: ',n_files)
    print('No of class: ', dirIndex)  ## no of classes
    no_of_class = dirIndex
    return image_list, label_list, class_list