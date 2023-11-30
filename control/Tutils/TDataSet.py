# -*-coding: utf-8 -*-
"""
    This program is modified based on  ref: https://blog.csdn.net/guyuealian/article/details/88343924
    @Objective: set up pytorch image data in mini_batch with data balancing
                add in create the image_label_list by given directory
    @File   : TDataset.py
    @Author : Tien
    @E-mail : fctien@ntut.edu.tw
    @Date   : 20200515
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_utils import image_processing
import os
 
class TDataset(Dataset):
    def __init__(self, image_dir, resize_height=256, resize_width=256, repeat=1, isBalancing = True):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.class_list = None
        self.isBalancing = isBalancing
        self.image_label_list = self.read_file(image_dir)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        
        
        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        self.toTensor = transforms.ToTensor()
 
        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        # self.normalize=transforms.Normalize()
 
    def __getitem__(self, i): 
        ## 當使用此object時，就像使用list or dict 一樣可以直接用i 取得對應之值
        ## 這裡是回傳 image 及 label
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_name, label = self.image_label_list[index]
        #image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_name, self.resize_height, self.resize_width, normalization=False)
        img = self.data_preproccess(img)
        label=np.array(label)
        return img, label
 
    def __len__(self): ## 
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len
 
    def read_file(self, image_dir):
        image_label_list = []
        No_Image_in_dirs, self.class_list = image_processing.find_no_image_in_dir(image_dir)
        import os  
        # ## read all images in training directory
        dirIndex = 0
        dirs = os.listdir(image_dir)
        #image_list = list()
        #label_list = list()
        for dir in dirs:
            fullpath = os.path.join(image_dir, dir)
            if os.path.isdir(fullpath):
                #files_path = os.path.join(fullpath, '*.jpg')
                #files = sorted(glob(files_path))
                files = image_processing.findAllImagFiles(fullpath)
                if self.isBalancing:
                    no_of_copy = int(max(No_Image_in_dirs) / No_Image_in_dirs[dirIndex]+0.5)
                else:
                    no_of_copy = 1
                #print("No of copies: ", no_of_copy)        
                no_of_image = 0
                for f in files:
                    for i in range(no_of_copy): ## data balancing
                        #img = load_img(f, target_size=(self.resize_width, self.resize_height, self.no_of_channel)) ## keras read data and reshape
                        #img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), -1) #cvImg = cv2.imread(f, -1) #
                        #img = cv2.resize(img, (self.resize_height, self.resize_width))
                        #new_img = img_to_array(img) 
                        image_label_list.append((f, dirIndex))
                        #label_list.append(dirIndex)      
                        no_of_image +=1
            print(dir, ":", no_of_image)
            dirIndex +=1  ## store 0, 1, 2, 3, 4        
        del files
        n_files = len(image_label_list)
        print('No of images: ',n_files)
        print('No of class: ', dirIndex)  ## no of classes
        no_of_class = dirIndex
        # with open(filename, 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
        #         content = line.rstrip().split(' ')
        #         name = content[0]
        #         labels = []
        #         for value in content[1:]:
        #             labels.append(int(value))
        #         image_label_list.append((name, labels))
        return image_label_list
 
    def load_data(self, path, resize_height, resize_width, normalization):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = image_processing.read_image(path, resize_height, resize_width, normalization)
        return image
 
    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data) ## convert into 0-1
        return data

def main():
    image_dir='./train'
    train_data = TDataset( image_dir=image_dir, resize_height= 224, resize_width = 224,repeat=1)
    print(len(train_data))
    return

if __name__ == "__main__":
    main()