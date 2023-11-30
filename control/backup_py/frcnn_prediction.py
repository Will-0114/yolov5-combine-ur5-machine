## https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/
## Using frcnn + COCO model to predict image

import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import time
import cv2

def main2():
    from  pytorch_utils import pytorch_utility
    from pytorch_utils import frcnn_utils
    t_start = time.perf_counter()
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = pytorch_utility.load_full_model(filename="./model/frcnn_PowerRed.pth")
    if torch.cuda.is_available:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    #t_end = time.perf_counter()
    #print("Load model time: ", round((t_end - t_start), 3), " sec.")
    #t_start = time.perf_counter()
    model = model.to(device)
    t_end = time.perf_counter()
    print("Load model into cuda time: ", round((t_end - t_start), 3), " sec.")
    model.eval()
    
    file_list = pytorch_utility.findAllImagFiles(path = "./test_data/JPEGImages")
    class_list = ["background", "c1", "c2", "c3"]
    threshold = 0.5
    for f in file_list:
        img = cv2.imread(f, 1) # Read image with cv2
        #print("Image size: ", img.shape)
        img_small = cv2.resize(img, None, fx = 0.5, fy =0.5)
        frcnn_utils.object_detection_frcnn_cvImg(model, img_small, class_list = class_list, threshold=threshold, isShow =True)
    return

if __name__ == "__main__":
    main2()
