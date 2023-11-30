
## Evaluation: 無法正確計算
import os
os.environ["PYTORCH_JIT"]="0" ##
import numpy as np
import torch
from PIL import Image
######### For preprocessing ###################
import xml.etree.ElementTree as ET
import cv2
import Tutils.utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from Tutils.engine import train_one_epoch, evaluate
import pprint
import time
from torchvision import transforms as T

## for pyinstaller to work, add this to disable jit
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj    
import torch.jit
torch.jit.script_method = script_method 
torch.jit.script = script
########################################## 
##############################################################################
## Preprocssing:
## 1. 請使用者將影像放置於: ./train_data/JPEGImages 資料夾中
## 2. 請使用者將標記 xml 檔案放置於：./train_data/Annotations 資料夾中
## 3. 請使用者將標記 class.txt (分類的class 名稱) 檔案放置於：　./train_data
##    以上動作是為了讓程式可以自動找到相關的檔案
## Return: Image_list, target_list
## 並於 ./train_data 中儲存train_file.txt  (Yolo可以用)
##  ## Note: 必須提供 class.txt (LabelImg中所產生的xml檔案中是以 class名稱儲存)
##############################################################################
## 產生 annotation file，save as annotation_file

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

def read_bbox_label_list(imgs_list, classes_list, train_path = "./train_data/"):
    annotation_filename = train_path + "/annotation_file.txt"
    list_file = open(annotation_filename, 'w')
    labels_list = list()
    boxes_list = list()
    for idx, image_name in enumerate(imgs_list):
        #print(image_name)
        image_id =image_name.split(".")[0]
        list_file.write(train_path+'/JPEGImages/%s.jpg'%(image_id.strip('\n'))) ## image type is jpg
        boxes, labels = convert_annotation(image_id, list_file, classes_list, train_path)
        #Image_list.append(rgb_tensor)
        boxes_list.append(boxes)
        labels_list.append(labels)
        list_file.write('\n')
    list_file.close()
    return boxes_list, labels_list

def frcnn_data_preprocessing(train_path = "./train_data", classes_path = "./train_data/classes.txt",
                          imageSet_path = "./train_data/JPEGImages", annotation_file = "./train_data/train_files.txt'"):
    classes_list = read_classes(classes_path = "./train_data/classes.txt")
    #print("classes label: ", classes_list, "\n")

    ## find sorted image_list
    imgs_list = list(sorted(os.listdir(os.path.join(train_path, "JPEGImages"))))
    #print(imgs_list)

    #print("Creating annotation file... ")
    # image_ids = open(imageSet_path)
    list_file = open(annotation_file, 'w')
    target_list = list()
    Image_list = list()
    for idx, image_name in enumerate(imgs_list):
        #print(image_name)
        cvImg = cv2.imread(train_path+'/JPEGImages/'+image_name, 1) ## color image
        rgb_img = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img/255.  ## to floating
        rgb_img = np.moveaxis(rgb_img, -1, 0)  ## switch to (c, w, h) format
        rgb_tensor = torch.from_numpy(rgb_img)
        rgb_tensor = rgb_tensor.float()  ## 改成tensor
        image_id =image_name.split(".")[0]
        list_file.write(train_path+'/JPEGImages/%s.jpg'%(image_id.strip('\n'))) ## image type is jpg
        box_list, label_list = convert_annotation(image_id, list_file, classes_list, train_path)
        Image_list.append(rgb_tensor)
        list_file.write('\n')
        ## prepare target_list
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        labels = torch.as_tensor(label_list, dtype=torch.int64)
        ## using cocoapi must add this dict: image_id, area, iscrowd
        image_id = torch.tensor([idx])  
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        ## Target:dict for using cocoapi's evaluate
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #print(labels)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target_list.append(target)  ## target is a list of dict, which are "boxes" and "labels" corrrespond with tensor of lists 
    list_file.close()
    
    return Image_list, target_list, classes_list


## 搜尋class name，然後將其改為 1, 2, 3, ... 
def convert_annotation(image_id, list_file, classes, train_path):
    #print(image_id)
    #print(list_file)
    #print(classes)
    image_id = image_id.strip('\n')
    in_file = open(train_path+'/Annotations/%s.xml'%(image_id), encoding="utf-8")
    #print("in file = ", in_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    box_list = list()
    label_list = list()
    for obj in root.iter('object'):  
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls) +1 ## class label start from 1 (cocoapi中的限制)??
        xmlbox = obj.find('bndbox')
        box = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        ## must check if in (xmin, ymin, xmax, ymax)
        [xmin, ymin, xmax, ymax] = box
        # if xmin == 916:
        #     print("find")
        # if xmax == 187:
        #     print("find")
        #     input()
        if xmax < xmin or ymax < ymin:
            xmin_new = min(xmin, xmax)
            xmax_new = max(xmin, xmax)
            ymin_new = min(ymin, ymax)
            ymax_new = max(ymin, ymax)
            box =[xmin_new, ymin_new, xmax_new, ymax_new]
        list_file.write(" " + ",".join([str(a) for a in box]) + ',' + str(cls_id))
        #box = torch.as_tensor(b, dtype=torch.float32) ## tensor轉換在最後list建好後
        #labels = torch.ones((,), dtype=torch.int64)
        box_list.append(box)
        #target["labels"] = cls_id #torch.as_tensor(cls_id, dtype =torch.int64)
        label_list.append(cls_id)
        #box_label_list.append(target)
        #print(box_list)
        #print(label_list)
    return box_list, label_list
## 目前沒有用到
def create_train_image_list(src, dest):
    print("Create image list for training...")
    if not os.path.isdir(os.path.dirname(dest)):
        os.mkdir(os.path.dirname(dest))
    out_file = open(dest,'w')  #生成了在指定目录下的txt文件  
    with open(dest, 'w') as f:
        for name in os.listdir(src):
            base_name = os.path.basename(name)
            file_name = base_name.split('.')[0]
            f.write('%s\n' % file_name)
    return

def read_annotation(image_id, classes, train_path):
    #print(image_id)
    #print(list_file)
    #print(classes)
    image_id = image_id.strip('\n')
    in_file = open(train_path+'/Annotations/%s.xml'%(image_id), encoding="utf-8")
    #print("in file = ", in_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    box_list = list()
    label_list = list()
    for obj in root.iter('object'):  
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls) +1 ## class label start from 1 (cocoapi中的限制)??
        xmlbox = obj.find('bndbox')
        box = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        ## must check if in (xmin, ymin, xmax, ymax)
        [xmin, ymin, xmax, ymax] = box
        # if xmin == 916:
        #     print("find")
        # if xmax == 187:
        #     print("find")
        #     input()
        if xmax < xmin or ymax < ymin:
            xmin_new = min(xmin, xmax)
            xmax_new = max(xmin, xmax)
            ymin_new = min(ymin, ymax)
            ymax_new = max(ymin, ymax)
            box =[xmin_new, ymin_new, xmax_new, ymax_new]
        #list_file.write(" " + ",".join([str(a) for a in box]) + ',' + str(cls_id))
        #box = torch.as_tensor(b, dtype=torch.float32) ## tensor轉換在最後list建好後
        #labels = torch.ones((,), dtype=torch.int64)
        box_list.append(box)
        #target["labels"] = cls_id #torch.as_tensor(cls_id, dtype =torch.int64)
        label_list.append(cls_id)
        #box_label_list.append(target)
        #print(box_list)
        #print(label_list)
    return box_list, label_list

def read_image_list(img_path = "./train_data/JPEGImages"):
    imgs_list = list(sorted(os.listdir(img_path)))
    return imgs_list

class TFRCNN_Dataset(object):
    def __init__(self, imgs_list, boxes_list, labels_list, classes_list, root="/train_data", transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = imgs_list #list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        #print(self.imgs)
        ## 取得class 名稱，read_annotations() 要用到
        self.classes_list = classes_list #read_classes(classes_path = self.root+ "/" + class_file)
        self.boxes_list = boxes_list
        self.labels_list = labels_list
        self.classes_file = root + "/" + "classes.txt"
        self.imageSet_path = root+ "/" + "JPEGImages"
        #annotation_file = root + "/" + "train_files.txt"

    def __getitem__(self, idx):
        # # load images
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        #img = Image.open(img_path).convert("RGB") ## load image with pil
        cvImg = cv2.imread(img_path, 1) ## color image
        rgb_img = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img/255.  ## to floating
        rgb_img = np.moveaxis(rgb_img, -1, 0)  ## switch to (c, w, h) format
        rgb_tensor = torch.from_numpy(rgb_img)
        img = rgb_tensor.float()  ## 改成tensor
        # # note that we haven't converted the mask to RGB,
        # # because each color corresponds to a different instance
        # # with 0 being background  ## class 0: 是背景
        image_id =self.imgs[idx].split(".")[0] ## find image base name
        #box_list, label_list = read_annotation(image_id, self.classes_list, self.root)
        #print(box_list, label_list)
        # # convert everything into a torch.Tensor
        ## boxes_list: 是依照 imgs_list 順序所儲存的 bbox list
        ## 減少磁碟讀取的次數 (前提: data bbox 數量不會超過 ram 讀取的量)
        boxes = torch.as_tensor(self.boxes_list[idx], dtype=torch.float32)
        labels = torch.as_tensor(self.labels_list[idx], dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        ## coco api 需要以下之格式
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

## 因為將所有的影像都load 進此class，所以暫時不用此class
# class FRCNN_Dataset(object):
#     def __init__(self, root="/train_data", transforms=None):
#         self.root = root
#         self.transforms = transforms
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
#         # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
#         classes_file = root + "/" + "classes.txt"
#         imageSet_path = root+ "/" + "JPEGImages"
#         annotation_file = root + "/" + "train_files.txt"
#         self.img_list, self.target_list, self.class_list =frcnn_data_preprocessing(train_path = root, classes_path =classes_file,
#                           imageSet_path = imageSet_path, annotation_file = annotation_file)

#     def __getitem__(self, idx):
#         # # load images ad masks
#         # img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
#         # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
#         # img = Image.open(img_path).convert("RGB")
#         # # note that we haven't converted the mask to RGB,
#         # # because each color corresponds to a different instance
#         # # with 0 being background
#         # mask = Image.open(mask_path)
#         # # convert the PIL Image into a numpy array
#         # mask = np.array(mask)
#         # # instances are encoded as different colors
#         # obj_ids = np.unique(mask)
#         # # first id is the background, so remove it
#         # obj_ids = obj_ids[1:]

#         # # split the color-encoded mask into a set
#         # # of binary masks
#         # masks = mask == obj_ids[:, None, None]

#         # # get bounding box coordinates for each mask
#         # num_objs = len(obj_ids)
#         # boxes = []
#         # for i in range(num_objs):
#         #     pos = np.where(masks[i])
#         #     xmin = np.min(pos[1])
#         #     xmax = np.max(pos[1])
#         #     ymin = np.min(pos[0])
#         #     ymax = np.max(pos[0])
#         #     boxes.append([xmin, ymin, xmax, ymax])

#         # # convert everything into a torch.Tensor
#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # # there is only one class
#         # labels = torch.ones((num_objs,), dtype=torch.int64)
#         # masks = torch.as_tensor(masks, dtype=torch.uint8)

#         # image_id = torch.tensor([idx])
#         # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # # suppose all instances are not crowd
#         # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         # target = {}
#         # target["boxes"] = boxes
#         # target["labels"] = labels
#         # target["masks"] = masks
#         # target["image_id"] = image_id
#         # target["area"] = area
#         # target["iscrowd"] = iscrowd
#         img, target = self.img_list[idx], self.target_list[idx]
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)

#         return img, target

#     def __len__(self):
#         return len(self.img_list)

def save_full_model(model, filename = "./model/ResNet_best"):
    torch.save(model, filename)  ## save entire model
    return

def load_full_model(filename="./model/Best_ResNet.pth"):
    try:
        model = torch.load(filename)
    except:
        print("Please make sure the model file exists")
    return model

## training frcnn: 
def train_frcnn(no_epoch = 100, lr = 0.01, num_classes = 4, batch_size = 2, train_data_path = "./train_data", 
             test_data_path = "./train_data", save_model_name ="./model/frcnn.pth", evaluate_period = 5 ):
    #frcnn_data_preprocessing(train_path = "./train_data", classes_path = "./train_data/classes.txt",
    #                      imageSet_path = "./train_data/JPEGImages", annotation_file = "./train_data/train_files.txt'")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Loading data...")
    ## read in the classes list and image list from training folder 
    class_file = "classes.txt" 
    classes_list = read_classes(classes_path = train_data_path +"/" + class_file) ## 減少dataset reading 的負擔
    imgs_list = read_image_list(img_path = train_data_path +"/JPEGImages")
    #print(imgs_list)
    boxes_list, labels_list = read_bbox_label_list(imgs_list, classes_list, train_path = train_data_path)
    #print(boxes_list)
    #print(labels_list)
    batch_size = batch_size
    dataset = TFRCNN_Dataset(boxes_list =boxes_list, labels_list =labels_list, imgs_list = imgs_list, classes_list = classes_list, root= train_data_path, transforms=None)
    dataset_test = TFRCNN_Dataset(boxes_list =boxes_list, labels_list =labels_list, imgs_list = imgs_list, classes_list = classes_list, root= train_data_path, transforms=None)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.RandomSampler(dataset_test)
    # #dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    #test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=0,
        collate_fn=pytorch_utils.utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=0,
        collate_fn=pytorch_utils.utils.collate_fn)
    print("Creating model")
    num_classes = num_classes  # 1 class (person) + background
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    ## 修改架構 : num_classes，其他相同
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #model = load_full_model(filename= load_model_name)
    model.to(device)
    print("Model created successfully")
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr= lr, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # let's train it for 10 epochs
    num_epochs = no_epoch
    print("Starting training process...")
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        if epoch  % evaluate_period == 0:
            ## 經過 eavluate 後，dataset boxes 會改變，只好重新load 進來
            dataset = TFRCNN_Dataset(boxes_list =boxes_list, labels_list =labels_list, 
                    imgs_list = imgs_list, classes_list = classes_list, root= train_data_path, transforms=None)
    
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1,
                sampler=test_sampler, num_workers=0,
                collate_fn=pytorch_utils.utils.collate_fn)           
            #print_dataloader_target(data_loader_test, key = "boxes")
            ## 儲存目前的 models
            t_start=time.perf_counter()
            save_full_model(model, filename = save_model_name)
            t_end =time.perf_counter()
            print("Saving model spent: ", round((t_end-t_start),3), " sec.")
            coco_evaluator = evaluate(model, data_loader_test, device=device) ## 不可以用training data_loader，training data 會出錯
            #print(coco_evaluator)
    save_full_model(model, filename = save_model_name)
    print("Training is done!")
    return model

def print_dataloader_target(dataloader, key = "boxes"):
    ## print target
    for i, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
            #inputs = Variable(inputs)
            #targets = Variable(targets)
            print("targets.data", targets[0][key])
    return

def retrain_frcnn(no_epoch = 100, lr = 0.01, batch_size = 2, train_data_path = "./train_data", test_data_path = "./train_data", 
                load_model_name = "./model/frcnn.pth", save_model_name ="./model/frcnn.pth",
                evaluate_period = 5 ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Loading data...")
    batch_size = batch_size
    ## read in the classes list and image list from training folder 
    class_file = "classes.txt" 
    classes_list = read_classes(classes_path = train_data_path +"/" + class_file) ## 減少dataset reading 的負擔
    imgs_list = read_image_list(img_path = train_data_path +"/JPEGImages")
    #print(imgs_list)
    boxes_list, labels_list = read_bbox_label_list(imgs_list, classes_list, train_path = train_data_path)
    #print(boxes_list)
    #print(labels_list)
    dataset = TFRCNN_Dataset(boxes_list =boxes_list, labels_list =labels_list, imgs_list = imgs_list, classes_list = classes_list, root= train_data_path, transforms=None)
    dataset_test = TFRCNN_Dataset(boxes_list =boxes_list, labels_list =labels_list, imgs_list = imgs_list, classes_list = classes_list, root= train_data_path, transforms=None)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.RandomSampler(dataset_test)
    # #dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=0,
        collate_fn=pytorch_utils.utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=0,
        collate_fn=pytorch_utils.utils.collate_fn)
    
    print("Creating model")
    # ## 不用修改架構 : 因為load完整 structure
    model = load_full_model(filename= load_model_name)
    model.to(device)
    print("Model created successfully")
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # let's train it for 10 epochs
    num_epochs = no_epoch
    print("Starting re-training process...")
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset using cocoapi
        
        if epoch  % evaluate_period == 0:
            ## 經過 eavluate 後，dataset boxes 會改變，只好重新load 進來
            dataset_test = TFRCNN_Dataset(boxes_list =boxes_list, labels_list =labels_list, imgs_list = imgs_list, 
                            classes_list = classes_list, root= train_data_path, transforms=None)
            data_loader_test = torch.utils.data.DataLoader(
                 dataset_test, batch_size=1,
                 sampler=test_sampler, num_workers=0,
                 collate_fn=pytorch_utils.utils.collate_fn)
            
            #print_dataloader_target(data_loader_test, key = "boxes")
            coco_evaluator = evaluate(model, data_loader_test, device=device) ## 不可以用training data_loader，training data 會出錯
            #pprint.pprint(coco_evaluator.__dict__)
            ## 儲存目前的 models
            t_start=time.perf_counter()
            save_full_model(model, filename = save_model_name)
            t_end =time.perf_counter()
            print("Saving model spent: ", round((t_end-t_start),3), " sec.")
    save_full_model(model, filename = save_model_name)
    print("Training is done!")
    return model

## Prediction of the model
def frcnn_prediction_cvImg(model, cvImg, threshold, class_list):
    #img = Image.open(img_path)
    img = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    #img.show()
    transform = T.Compose([T.ToTensor()])
    img = transform(img) # Apply the transform to the image
    img = img.cuda()  ## convert into cuda()  GPU
    pred = model([img]) # Pass the image to the model
    ## remember: convert pred contain back to cpu  by tensor.cpu()
    #print(type(pred), pred)
    pred_class = [class_list[i] for i in list(pred[0] ['labels'].cpu().numpy())] # Get the Prediction Score
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
        pred_boxes = pred_boxes[:pred_t+1]  ## after threshold
        pred_class = pred_class[:pred_t+1]  ## after threshold 14 left
    except:
        pred_boxes =[]
        pred_class =[]
    return pred_boxes, pred_class, pred_score

## must use GPU, otherwise prediction take more than 8 seconds.
def object_detection_frcnn_cvImg(model, cvImg, class_list, threshold=0.5, rect_th=2, text_size=1, text_th=2,  isShow=False):
    img = cvImg.copy()
    t_start = time.perf_counter()
    #boxes, pred_cls, pred_score = get_prediction(model, img_path, threshold) # Get predictions
    boxes, pred_cls, pred_score = frcnn_prediction_cvImg(model, img, threshold, class_list) # Get predictions
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size,  (0,255,0), thickness=text_th) # Write the prediction class
    t_end = time.perf_counter()
    print("Pred time: ", round((t_end - t_start), 3), " sec.")
    if isShow:
        cv2.imshow("Image:", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plt.figure(figsize=(20,30)) # display the output image
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
    return boxes, pred_cls, pred_score, img

def object_detection_frcnn_cvImg_mbbox(model, cvImg, class_list, threshold=0.5, rect_th=2, 
                                    min_BBox = (10, 10), text_size=1, text_th=2,  isShow=False):
    img = cvImg.copy()
    t_start = time.perf_counter()
    #boxes, pred_cls, pred_score = get_prediction(model, img_path, threshold) # Get predictions
    boxes, pred_cls, pred_score = frcnn_prediction_cvImg(model, img, threshold, class_list) # Get predictions
    
    ## add screening 
    screened_boxes = list()
    screened_pred_cls = list()
    screened_pred_score = list()
    for i, box in enumerate(boxes):
        width = box[1][0] - box[0][0]  ## point(x, y) = point(col, row)
        height = box[1][1] - box[0][1] ## box( (x1, y1), (x2, y2))
        print(height, width)
        if (height > min_BBox[0]) or (width > min_BBox[1]):
            screened_boxes.append(box)
            screened_pred_cls.append(pred_cls[i])
            screened_pred_score.append(pred_score[i])

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(screened_boxes)):
        cv2.rectangle(img, screened_boxes[i][0], screened_boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img, screened_pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size,  (0,255,0), thickness=text_th) # Write the prediction class
    t_end = time.perf_counter()
    print("Pred time: ", round((t_end - t_start), 3), " sec.")
    if isShow:
        cv2.imshow("Image:", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plt.figure(figsize=(20,30)) # display the output image
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
    return screened_boxes, screened_pred_cls, screened_pred_score, img

if __name__ == "__main__":
    retrain_frcnn(lr = 0.01, evaluate_period = 1) ## add parameter: data_path, training params, save model name
    #train_frcnn(no_epoch = 100, num_classes = 4, batch_size = 2, train_data_path = "./train_data", 
    #         test_data_path = "./train_data", save_model_name ="./model/frcnn_1.pth", evaluate_period = 1 )
