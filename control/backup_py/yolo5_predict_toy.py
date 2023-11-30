from Tutils.yolov5_utils import yolov5_setup, yolov5_load_model, yolov5_predict
import cv2
from Tutils.tien_utility import findAllImagFiles, split_filename_extension
from Tutils.Draw_Utility import draw_rectangle

def read_bbox_list_yolo(filename): ## read wise format
    f = open(filename, "r")
    lines = f.readlines()
    bbox_list = list()
    for line in lines:
        data = line.split(" ")
        tag = data[0]
        #desc = data[1]
        bbox = (float(data[1]), float(data[2]), float(data[3]), float(data[4]))
        bbox_list.append(bbox)
    return bbox_list

if __name__ == '__main__':
    yolov5_setup()
    img_size = 1024
    conf_thres = 0.7
    iou_thres = 0.3
    model, device, half, class_names, colors = yolov5_load_model(model_path= "./weights/best.pt", imgsz = img_size)
    #model.eval()
    path = "./inference/images/toys" #zidane.jpg"
    img_list = findAllImagFiles(path)
    for fn in img_list:
        img0 = cv2.imread(fn, -1)  # BGR
        row, col, channel = img0.shape
        dir, filename, ext = split_filename_extension(fn)
        class_names =["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"]     
        # class_names =["0","1"]  
        #cvGoodImage = cv2.resize(self.cvGoodImage, (0, 0), fx= self.scale, fy=self.scale)
        assert img0 is not None, 'Image Not Found ' + path
        display_img, det_list = yolov5_predict(model, img0, device, names=class_names, half=half, img_size = img_size, conf_thres = conf_thres, iou_thres = iou_thres, view_img= False, isRandomColor=False)
        bbox_list = read_bbox_list_yolo(filename = dir + "/" + filename + ".txt")
        ## in yolo form centerx, centery, width, height (normalized)
        # for bbox in bbox_list:
        #     p1 = (int((bbox[0]-bbox[2]/2)*col), int((bbox[1]-bbox[3]/2)*row))
        #     p2 = (int((bbox[0]+bbox[2]/2)*col), int((bbox[1]+bbox[3]/2)*row))
        #     display_img = draw_rectangle(display_img, p1= p1, p2 = p2, lineWidth=2, color =(0, 255, 0)) ## draw green
        #print(det_list) ## [x1, y1, x2, y2, conf, class]
        out_fn = "./output/out_"+ filename + ext #"./inference/output/E_Coli_Predict/out_"+ filename + ext
        #out_fn = out_fn.replace(",", "_")
        cv2.imwrite(out_fn, display_img)
        cv2.imshow("yolo v5 Capacitor:", display_img)
        cv2.waitKey(0)