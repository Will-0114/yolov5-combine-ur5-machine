import os
from glob import glob
import datetime
import winsound
import cv2
import Tutils.DIP_Class
import shutil

def convert_cv2_2_pil(cvImg):
    # You may need to convert the color.
    img = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

def read_classes(classes_path = "./train_data/classes.txt"):
    #print("Creating classes label... ")
    class_ids = open(classes_path)
    classes_list = list()
    for cls_id in class_ids:
        cls_id = cls_id.strip('\n')
        #print(cls_id)
        classes_list.append(cls_id)
    return classes_list

def write_clases(classes_list, path):
    filename = path + "\\classes.txt"
    with open(filename, 'w', encoding='utf8') as f:
        for l in classes_list:
            f.write(l + "\n")
    # with open('./model/class_list.txt', 'w', encoding='utf8') as f:
    #     for l in class_list:
    #         f.write(l + "\n")
    return

def nothing(x):
  pass

## find filename and ext
def split_filename_extension(full_filename):
    path_name = os.path.dirname(full_filename)
    filename_w_ext = os.path.basename(full_filename)
    filename, file_extension = os.path.splitext(filename_w_ext)
    return path_name, filename, file_extension

def find_jpg(path):
    pattern = os.path.join(path, '*.jpg')
    jpg_files = sorted(glob(pattern))
    #print(jpg_files)
    return jpg_files

## 20180517  新增函數: 搜尋資料夾下所有影像檔
def findAllImagFiles(path = "./train/"):  ## 搜尋目錄下所有相關影像之檔名
    pattern = os.path.join(path, '*.bmp') 
    bmp_files = sorted(glob(pattern))
    #print(type(bmp_files))
    pattern = os.path.join(path, '*.jpg')
    jpg_files = sorted(glob(pattern))
    pattern = os.path.join(path, '*.png')
    png_files = sorted(glob(pattern))
    file_list = bmp_files + jpg_files + png_files
    return file_list  ## 回傳檔名的 list

def findAllImages(path = "./train/"):
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

## read feature from a text file which features are splited by ","    
def read_csv(fn = "./export/d5_DPS-3000FB A/DPS-3000FB A_20180422.txt"):
    text_file = open(fn, "r")
    lines = text_file.read().split('\n')
    feature_list= list()
    label_list = list()
    f_list = list()
    for l in lines:
        #print(l)        
        feature = l.split(',')
        print(feature)
        feature_list.append(feature)
    return feature_list

from glob import glob
def findAllFiles(path):
        pattern = os.path.join(path, '*.*') 
        all_files = sorted(glob(pattern))
        return all_files


def findAllFiles(path, file_type = "*.txt"):
    pattern = os.path.join(path, file_type) 
    all_files = sorted(glob(pattern))
    return all_files

def findAllDir(path):
    """
    Find all folder under path
    return the list of folder name
    """
    if not os.path.exists(path):
        print("Current directory does not exist:", path)
        return None
    #dir_list = [os.path.basename(x) for x in filter(os.path.isdir, glob.glob(os.path.join(path, '*')))]
    dir_list = [ f.path for f in os.scandir(path) if f.is_dir() ]
    dirname_list=list()
    for dir in dir_list:
        dir = dir.replace("/", "\\")
        dirname_list.append(dir)
    ##print(dirname_list)
    return dirname_list

def findAllDirName(path):
    """
    Find all folder under path
    return the list of folder name (not with path)
    """
    dirname_list = list()
    dir_list = findAllDir(path)
    for dir in dir_list:
        dir = dir.replace("/", "\\")
        d_list = dir.split("\\")
        n = len(d_list)
        #print(d_list[n-1])
        dirname_list.append(d_list[n-1])
    return dirname_list

def write_class(class_list, path="./", filename="class.txt"):
    f = open(path + "/" + filename, "w")
    for c in class_list:
        f.write(c+"\n")
    f.close()
    return

## 將目錄下所有的影像都改為彩色影像 (如果式灰階)
def ConvertAllImageToColor(path = ".\\train\\original"):
    files = findAllImages("./Train/Original")
    for f in files:
        img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), 1)  ## read as color
        cv2.imencode('.bmp', img)[1].tofile(f) 
    return

def get_today():
    #today = date.today()
    #d = datetime.datetime.today()
    d_str = datetime.datetime.now().strftime("%Y_%m_%d") #("%d/%m/%Y") #("%d/%m/%Y")
    print("Today's date:", d_str)
    return d_str

def get_now():
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%h_%s") #("%d/%m/%Y") #("%d/%m/%Y")
    print("Current time:", time_str)
    return time_str

## 20181120 beep sound created by windows:　　Must import winsound
def beep(dureation = 500):
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    return

### Function: add prefix G_ or NG_ to all the file in dir
def rename(directory, prefix):
    pattern = os.path.join(directory, '*.*') ### can change any other file pattern
    _files = sorted(glob(pattern))
    for fn in _files:
        ##print(fn)
        title, ext = os.path.splitext(os.path.basename(fn))
        ##print(title)
        new_filename = os.path.join(directory, prefix + title +ext)
        if not os.path.exists(new_filename):
            ##print(new_filename)
            os.rename(fn, new_filename)

def read_data_into_list(fn = "./data.txt"):
    text_file = open(fn, "r")
    lines = text_file.read().split('\n') # 一行
    data_list= list()
    for l in lines:
        #print(l)
        if l == "":
            continue  ##skip the empty line
#        f = l.split(',' , '\t', '')
        f = re.split('; |, |\*|\n', l)
        #print(f)
        f_list=list()
        for i in range(len(f)):
            f_list.append(float(f[i]))
        
        data_list.append(f_list)
    #print(data_list)
    print("no of data: ", len(data_list))
    return data_list  ## return data as list of list

def write_to_text(feature_list, fn = "f.txt"):
    with open(fn, 'w') as f:
        for item in feature_list:
            #print(len(item))
            #print(type(item))
            if type(item) == str:
                f.write("%s\n" % item)
            else:  ## list of list               
                for i in range(len(item)):
                    f.write("%s\t" % item[i])
                f.write("\n")    
    return

## image processing: threshold with trackbar
def threshold_trackbar(img, initial = 51, scale = 0.5):
    cv2.namedWindow('Colorbars')
    hh='Min'
    hl='Max'
    wnd = 'Colorbars'
    cv2.createTrackbar("Min", "Colorbars", 0, 255, nothing)
    cv2.createTrackbar("Max", "Colorbars", 255, 255, nothing) ## pre-set 255
    cv2.setTrackbarPos("Min", "Colorbars", initial) ## set initial value
    #img = cv2.imread('plastic3.jpg',0)
    imgShow = cv2.resize(img, (0,0), fx=scale, fy=scale)
    # titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    # images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    # for i in xrange(6):
    #     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]),plt.yticks([])
    # plt.show()
    while(1):
       hul=cv2.getTrackbarPos("Min", "Colorbars")
       huh=cv2.getTrackbarPos("Max", "Colorbars")
       ret,thresh1 = cv2.threshold(img, hul, huh, cv2.THRESH_BINARY)
       ret,threshShow = cv2.threshold(imgShow, hul, huh, cv2.THRESH_BINARY)
       #ret,thresh2 = cv2.threshold(img,hul,huh,cv2.THRESH_BINARY_INV)
       #ret,thresh3 = cv2.threshold(img,hul,huh,cv2.THRESH_TRUNC)
       #ret,thresh4 = cv2.threshold(img,hul,huh,cv2.THRESH_TOZERO)
       #ret,thresh5 = cv2.threshold(img,hul,huh,cv2.THRESH_TOZERO_INV)
       # cv2.imshow(wnd)
       cv2.imshow("Segmented Image (ESC to quit)",threshShow)
       #cv2.imshow("thresh2",thresh2)
       #cv2.imshow("thresh3",thresh3)
       #cv2.imshow("thresh4",thresh4)
       #cv2.imshow("thresh5",thresh5)
       k = cv2.waitKey(1) & 0xFF
       if k == ord('m'):
         mode = not mode
       elif k == 27:
         break
    cv2.destroyAllWindows()
    return hul, huh, thresh1

def color_threshold_slider(img):
    # named ites for easy reference
    barsWindow = 'Bars'
    hl = 'H Low'
    hh = 'H High'
    sl = 'S Low'
    sh = 'S High'
    vl = 'V Low'
    vh = 'V High'

    # set up for video capture on camera 0
    #cap = cv.VideoCapture(0)

    # create window for the slidebars
    cv.namedWindow(barsWindow, flags = cv.WINDOW_AUTOSIZE)

    # create the sliders
    cv.createTrackbar(hl, barsWindow, 0, 179, nothing)
    cv.createTrackbar(hh, barsWindow, 0, 179, nothing)
    cv.createTrackbar(sl, barsWindow, 0, 255, nothing)
    cv.createTrackbar(sh, barsWindow, 0, 255, nothing)
    cv.createTrackbar(vl, barsWindow, 0, 255, nothing)
    cv.createTrackbar(vh, barsWindow, 0, 255, nothing)

    # set initial values for sliders
    cv.setTrackbarPos(hl, barsWindow, 0)
    cv.setTrackbarPos(hh, barsWindow, 179)
    cv.setTrackbarPos(sl, barsWindow, 30)
    cv.setTrackbarPos(sh, barsWindow, 255)
    cv.setTrackbarPos(vl, barsWindow, 0)
    cv.setTrackbarPos(vh, barsWindow, 255)

    while(True):
        #ret, frame = cap.read()
        frame = cv.GaussianBlur(img, (5, 5), 0)
        
        # convert to HSV from BGR
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # read trackbar positions for all
        hul = cv.getTrackbarPos(hl, barsWindow)
        huh = cv.getTrackbarPos(hh, barsWindow)
        sal = cv.getTrackbarPos(sl, barsWindow)
        sah = cv.getTrackbarPos(sh, barsWindow)
        val = cv.getTrackbarPos(vl, barsWindow)
        vah = cv.getTrackbarPos(vh, barsWindow)

        # make array for final values
        HSVLOW = np.array([hul, sal, val])
        HSVHIGH = np.array([huh, sah, vah])

        # apply the range on a mask
        mask = cv.inRange(hsv, HSVLOW, HSVHIGH)
        maskedFrame = cv.bitwise_and(frame, frame, mask = mask)

        # display the camera and masked images
        cv.imshow('Masked Segmented (ESC to quit)', maskedFrame)
        #cv.imshow('Camera', frame)

            # check for q to quit program with 5ms delay
        k = cv.waitKey(1) & 0xFF
        if k & 0xFF == 27: ##ord('m'):
            break
    cv.destroyAllWindows()
    return hul, huh, sal, sah, val, vah, maskedFrame


def segment_color_image(Img, hul, huh, sal, sah, val, vah):
    # make array for final values
    frame = cv.GaussianBlur(Img, (5, 5), 0)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    HSVLOW = np.array([hul, sal, val])
    HSVHIGH = np.array([huh, sah, vah])
    mask = cv.inRange(hsv, HSVLOW, HSVHIGH)
    maskedImg = cv.bitwise_and(Img, Img, mask = mask)
    grayImg = cv.cvtColor(maskedImg, cv.COLOR_BGR2GRAY)
    
    #ret, binaryImg = cv.threshold(grayImg, 255, 25, cv.THRESH_BINARY)
    #binary_img = cv.cvtColor(binaryImg, cv.COLOR_GRAY2BGR)
    cv.imwrite("gray.jpg", grayImg)
    
    return grayImg

def append_filename_suffix(path, suffix = "d"):
    """
    將資料夾中所以檔案加入suffix
    """
    # 1. find all image name
    file_list = findAllImagFiles(path)
    for file in file_list:
        pathname, filename, file_extension = split_filename_extension(file)
        print(pathname, filename, file_extension)
        new_fn = pathname+ "/" + filename + "_" + suffix + file_extension
        print(new_fn)
        os.rename(file, new_fn)
    return 

def change_intensity_in_path(path = "./images_dark", value = -10):
    """
    將資料夾內的所有影像的亮度改變
    """
    dip = DIP_Class.DIP()
    file_list = findAllImagFiles(path)
    for file in file_list:
        cvImage = cv2.imread(file, -1)
        cvImage_d = dip.change_brightness(cvImage, -30)
        cv2.imwrite(file, cvImage_d)
        #cv2.imshow("original", cvImage)
        #cv2.imshow("Darken", cvImage_d)
        #cv2.waitKey(0)
    return

def copy_change_intensity(path_src= "./images", path_dest ="./images_dark", value = -13, suffix = "d"):
    ## 產生不同亮度之鋼珠##
    dip = DIP_Class.DIP()
    #suffix = "d"
    #path_src = "./images"
    #path_dest = "./images_dark"
    if not os.path.exists(path_dest):
        os.mkdir(path_dest)
    ## copy all image to dest
    file_list = findAllImagFiles(path_src)
    for file in file_list:
        pathname, filename, file_extension = split_filename_extension(file)
        print(pathname, filename, file_extension)
        new_fn = path_dest+ "/" + filename + "_" + suffix + file_extension
        cvImage = cv2.imread(file, -1)
        cvImage_d = dip.change_brightness(cvImage, value)
        cv2.imwrite(new_fn, cvImage_d)
    return
def fast_scandir(dir= "./config/products"):
    subfolders= [f.path for f in os.scandir(dir) if f.is_dir()]
    for dir in list(subfolders):
        subfolders.extend(fast_scandir(dir))
    for dir in subfolders:
        path=os.path.basename(dir)
        print(path)
    return subfolders

def fast_scandir_name(dir= "./config/products"):
    subfolders= [f.path for f in os.scandir(dir) if f.is_dir()]
    #for dir in list(subfolders):
    #    subfolders.extend(fast_scandir(dir))
    dir_name_list = list()
    dir_list = list()
    for dir in subfolders:
        dir_name_list.append(os.path.basename(dir))
        dir_list.append(os.path.dirname(dir))
    #print(dir_name_list)
    #print(dir_list)
    sub_dir_list = list()
    for i, dir in enumerate(dir_list):
        sub_dir_list.append(dir+ "/" +dir_name_list[i])
    #print(sub_dir_list)
    return sub_dir_list, dir_name_list

if __name__== "__main__":
    #print(find_jpg("C:/Users/User/Desktop/python_practice/crop_images"))
    #copy_change_intensity(path_src= "./Dest", path_dest ="./Dest_dark", value = -15, suffix = "d")
    class_list = findAllDirName("./training_data/face")
    print(class_list)
    write_class(class_list)