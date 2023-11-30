"""
    This utitlity program is written by FC Tien for teaching PyTorch
    @Objective: General utitlity program for ResNet_like classification function
                including: predict, load_full_model, predict....
    @File   : pytorch_utitlity.py
    @Author : FC. Tien (Dept. of IE&M, Taipei Tech)
    @E-mail : fctien@ntut.edu.tw
    @Date   : 20200515
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import numpy as np
#from Tutils.pytorch_utils import image_processing, TDataSet
import os
import torchvision.models as models  ## use build-in model
import time
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
os.environ["PYTORCH_JIT"]="0"

def predict_cvImg(model, device, cvImage, resize_height=224, resize_width =224, isShow=True):
    """
    Input: 
        model: ResNet (Use load_full_model to load and pass in this function)
        device: "cuda:0"
        cvImage: 輸入 cv2 影像，BGR numpy matrix
        Resize width & Height: ResNet default 224x224
    """
    cvImage = cv2.resize(cvImage, (resize_width, resize_height))
    rgb_img = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    ## 
    # transform2 = transforms.Compose([
    #     transforms.ToTensor(), 
	#  ]
    # )
    # img_cv_Tensor=transform2(rgb_img)
    # ## add unsqueeze() && switch RGB
    # #print(img_cv_Tensor.shape)
    # img_cv_Tensor=torch.unsqueeze(img_cv_Tensor, 0) 
    #inputs = img_cv_Tensor.to(device)
    model.eval()
    # do something like transform does  ## using Transform is faster? about the same
    rgb_img = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img/255.  ## to floating
    rgb_img = np.moveaxis(rgb_img, -1, 0)  ## switch to (c, w, h) format
    rgb_tensor = torch.from_numpy(rgb_img)
    rgb_tensor = rgb_tensor.float()
    rgb_tensor= torch.unsqueeze(rgb_tensor, 0) 
    #print(rgb_tensor.shape)
    inputs = rgb_tensor.to(device)
    outputs = model(inputs)
    ## calculate the prob.
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(outputs)
    _, predicted = outputs.max(1)
    label = predicted.item()
    prob = probabilities[0][label].item()
    #print("Label/Probl: ", label, prob)
    if isShow:
        cv2.imshow("image", cvImage)
        cv2.waitKey(0)
    return  label, prob

def predict(model, device, resize_height = 224, resize_width = 224, filename = "./test/01_Normal/001.bmp", isShow = True):
    normalization = False
    image = image_processing.read_image(filename, resize_height, resize_width, normalization)
    if isShow:
        cv2.imshow("image", image)
        cv2.waitKey(0)
    from torchvision import transforms
    # cvImg = cv2.imread(filename, -1)
    transform2 = transforms.Compose([
        transforms.ToTensor(), 
	 ]
    )
    img_cv_Tensor=transform2(image)
    ## add unsqueeze() && switch RGB
    #print(img_cv_Tensor.shape)
    img_cv_Tensor=torch.unsqueeze(img_cv_Tensor, 0) 
    #print(img_cv_Tensor.shape)
    model.eval()
    inputs = img_cv_Tensor.to(device)
    outputs = model(inputs)
    ## calculate the prob.
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(outputs)
    _, predicted = outputs.max(1)
    label = predicted.item()
    prob = probabilities[0][label].item()
    #print("Label/Probl: ", label, prob)
    return  label, prob

def load_full_model(filename="./model/Best_ResNet.pth"):
    model = torch.load(filename)
    return model

def test(epoch, net, testloader, device, criterion):
    #best_acc = 0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, dtype =torch.long)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item() 
            #print(epoch, "==> Loss: ", train_loss, " correct: ", round(correct/total*100, 4))
            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    return test_loss, acc

def save_full_model(model, filename = "./model/ResNet_best"):
    torch.save(model, filename)  ## save entire model
    return

def write_training_process(loss_list, filename="./model/train_process.txt"):
    with open(filename, "w") as f:
        for i in range(len(loss_list)):
            f.write(str(loss_list[i][0]) + ", " + str(loss_list[i][1]) + "\n")
    return

# callback loss-plot
def plot_loss(loss_list): ## loss_list  [(train_loss, test_loss)]
    import matplotlib.pyplot as plt
    plt.ion()
    plt.clf()
    plt.tight_layout()
    plt.title('The training process - Loss')
    loss = np.asarray(loss_list)[:, 0]
    val_loss = np.asarray(loss_list)[:, 1]
    x = [i for i in range(len(loss))]
    plt.plot(x, loss, color="red", marker = '.')
    plt.plot(x, val_loss, color = 'blue', marker = '.')
    plt.pause(1)
    plt.show()  
    if not os.path.isdir('./model'):
        os.mkdir('./model') 
    plt.savefig("./model/Loss_figure.png")
    return

# callback loss-plot
def plot_acc(acc_list):
    import matplotlib.pyplot as plt
    plt.ion()
    plt.clf()
    plt.tight_layout()
    plt.title('The training process - Accuracy')
    train_acc = np.asarray(acc_list)[:, 0]
    val_acc = np.asarray(acc_list)[:, 1]
    x = [i for i in range(len(train_acc))]
    plt.plot(x, train_acc, color="red", marker = '.')
    plt.plot(x, val_acc, color = 'blue', marker = '.')
    plt.pause(1)
    plt.show()   
    if not os.path.isdir('./model'):
        os.mkdir('./model')
    plt.savefig("./model/Accuracy_figure.png")
    return

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

def read_classes(classes_path = "./train_data/classes.txt"):
    #print("Creating classes label... ")
    class_ids = open(classes_path)
    classes_list = list()
    for cls_id in class_ids:
        cls_id = cls_id.strip('\n')
        #print(cls_id)
        classes_list.append(cls_id)
    return classes_list

def save_class_list(class_list):
    with open('./class_list.txt', 'w', encoding='utf8') as f:
        for l in class_list:
            f.write(l + "\n")
    # with open('./class_list.txt', 'w', encoding='utf8') as f:
    #     for l in class_list:
    #         f.write(l + "\n")
    return

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
    #print("Max No: ", max(No_Image_in_dirs))
    del file
    gc.collect()
    save_class_list(class_list)
    return No_Image_in_dirs, class_list

def evaluate_score(image_dir, model, save_path = "./Model/Misclassified"):  ## conduct the training/val images evaluation 9by batch
    """
    1. Calculate the confusion matrix by sklearn.metric
    2. Find all misclassified image and save into a save_path
    3. Calculate the precision, recall, F1 score
    """
    ## find all misclassified images in original training images
    import os
    import gc
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    print("[MSG]: Reading the data by classes with no data balancing ...")
    no_image_in_dirs, class_list = find_no_image_in_dir(path=image_dir)
    # ## read all images in training directory
    dirIndex = 0
    tag_list = list()
    predict_list = list()
    acc = 0
    dirs = os.listdir(image_dir)
    count = 0
    for dir in dirs:
        fullpath = os.path.join(image_dir, dir)
        if os.path.isdir(fullpath):
            #files_path = os.path.join(fullpath, '*.jpg')
            #files = sorted(glob(files_path))
            files = findAllImagFiles(fullpath)
            ## copy def read_data, but force no_of_copy = 1
            no_of_copy = 1 ##  int(max(No_Image_in_dirs) / No_Image_in_dirs[dirIndex]+0.5)
            #print("No of copies: ", no_of_copy)        
            no_of_image = 0           
            for f in files:
                #try:
                    for i in range(no_of_copy):
                        #img = load_img(f, target_size=(self.size_image_x, self.size_image_y, self.no_of_channel)) ## keras read data and reshape
                        img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), -1) #img = cv2.imread(f, -1)
                        #prob, tag = self.predit_cvImage(img)# replace this by pytorch predict_cvImg
                        tag, prob = predict_cvImg(model, device, img, resize_height =224, resize_width = 224, isShow=False)
                        #print(prob)
                        tag_list.append(tag)
                        #print(dirIndex)
                        predict_list.append(dirIndex)
                        count+=1
                        if tag == dirIndex:
                            acc = acc + 1
                            #print("correct")
                        else:
                            #print("Wrong")
                            #save_path = "./Misclassified/"
                            fn = os.path.basename(f)
                            filename  = save_path + "/" + class_list[dirIndex]+ "_to_" + class_list[tag] + "_" + fn
                            cv2.imwrite(filename, img)
                            #x = x.reshape((self.size_image_x, self.size_image_y, self.no_of_channel) )
                            #wrong_x_list.append(x)    
                # except:
                #     print("[MSG] Data reading error...", f)
                #     continue
                #    no_of_image +=1
            dirIndex +=1  ## store 0, 1, 2, 3, 4
            #print(dir, ":", no_of_image)
    print("Total number of image: ", count)
    acc = acc / count
    print("Over all ACC = ", acc)

    accuracy = accuracy_score(tag_list, predict_list)
    print("Accuracy = ", accuracy)
    recall_all_micro = recall_score(tag_list, predict_list, average='micro')
    print("Recall(mirco) = ", recall_all_micro)
    recall_all_macro = recall_score(tag_list, predict_list, average='macro')
    print("Recall (marco) = ", recall_all_macro)
    precision_all_macro = precision_score(tag_list, predict_list, average='macro')
    print("Precision (marco) = ", precision_all_macro)
    precision_all_micro = precision_score(tag_list, predict_list, average='micro')
    print("Precision (mirco) = ", precision_all_micro)
    f1_all_macro = f1_score(tag_list, predict_list, average='macro')
    print("F1 Score (macro) = ", f1_all_macro)
    f1_all_micro = f1_score(tag_list, predict_list, average='micro')
    print("F1 Score (micro) = ", f1_all_micro)
    cm_all = confusion_matrix(tag_list, predict_list )
    print(cm_all)
    fn = "result.txt"
    fn = save_path + "/" + fn
    f = open(fn, "w", encoding='utf8')
    f.write("Overall Acc = " + str(round(acc, 4)) + "\n" )
    f.write("Recall(micro) = " + str(recall_all_micro) +"\n")
    f.write("Recall(marco) = " + str(recall_all_macro)+"\n")
    f.write("Precision(micro) = " + str(precision_all_micro) +"\n")
    f.write("Precision(marco) = " + str(precision_all_macro)+"\n")
    f.write("F1 Score (micro) = " + str(f1_all_micro) +"\n")
    f.write("F1 Score (marco) = " + str(f1_all_macro)+"\n")
    f.write("[Result]\n")
    #f.write("Model = " + path + "\n")
    f.write("Confusion Matrix (Overall): \n")
    f.write(str(cm_all))
    f.close()
    return acc, cm_all, precision_all_micro, recall_all_micro, f1_all_micro

def train_ResNet(train_image_dir = "./train", val_image_dir = "./test", no_epachs = 100, lr = 0.01,
                    type = "ResNet18", pretrained = True, batch_size = 100, 
                    isDraw = True, model_name = "Best_ResNet.pth", split_ratio = 0.8):
    if not os.path.isdir("./model"):
        os.dir.mkdir("./model")
    ## check if gpu available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")
    print("Training ResNet using ", device)

    ## reset default image size = 224
    train_data = TDataSet.TDataset( image_dir=train_image_dir, resize_height= 224, resize_width = 224,repeat=1)
    total = len(train_data)
    num_classes = len(train_data.class_list)

    ## select model ## choose one
    if type == "ResNet18":
        model = models.resnet18(pretrained=pretrained)
    elif type == "ResNet34":
        model = models.resnet34(pretrained=pretrained)
    elif type == "ResNet50":
        model = models.resnet50(pretrained=pretrained)
    elif type == "ResNet101":
        model = models.resnet101(pretrained=pretrained)
    elif type == "ResNet152":
        model = models.resnet152(pretrained=pretrained)
    else:
        model = models.resnet18(pretrained=pretrained)
    ## Change the number_class feature ##############
    fc_features = model.fc.in_features
    #修改類別爲 num_classes = 6  Note: 修改架構必須在 model.to(device) 之前
    model.fc = nn.Linear(fc_features, num_classes)
    ################################################
    net = model.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    ## criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr= lr) #,
                      #momentum=0.8, weight_decay=5e-4)
    epoch_num= no_epachs   #总样本循环次数
    
    import math  ## split the data (Note: num_data must be large, even data balancing is done in TDataSet)
    train_data, test_data = random_split(train_data, [math.ceil(total*split_ratio), total -math.ceil(total*split_ratio)])
    print("Split train data into: ", len(train_data), ":",len(test_data))
    #val_data = TDataSet.TDataset( image_dir=val_image_dir, resize_height= 224, resize_width = 224,repeat=1)
    # for image, label in train_data:
    #      print(image)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size= batch_size, shuffle=False)
    
    #val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True) ## for validation
    loss_list = list()
    acc_list = list()
    print("Start training proces ...")
    best_acc = 0
    for epoch in range(epoch_num):
        print("Epoch: ", epoch +1)
        t_start = time.perf_counter()
        correct = 0
        total = 0
        train_loss = 0
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader): ## __getitem__
            #inputs, targets = Variable(inputs), Variable(targets)
            inputs, targets = inputs.to(device), targets.to(device, dtype = torch.long)
            optimizer.zero_grad()
            outputs = net(inputs)
            #print(outputs.shape)
            #print(targets.shape)
            loss = criterion(outputs, targets) ## outputs:[batch, 3, row, col], target: [batch]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print("Batch: ", batch_idx, "==>", end="") ## no effect
        
        train_acc = correct/total*100
        #print(train_loss, test_loss)
        t_end = time.perf_counter()
        print("==> Time span: ", round((t_end - t_start), 2), " sec.")
        print("==> Train Loss: ", round(train_loss, 4), " Train Acc: ", round(train_acc, 4))
        ############# testing #############
        test_loss, test_acc = test(epoch, net = net, testloader = test_loader, device = device, criterion = criterion)
        acc_list.append((train_acc, test_acc))
        loss_list.append((train_loss, test_loss))
        ## Save model ### Save checkpoint.
        if test_acc > best_acc:
            print('Saving best model..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            if not os.path.isdir('./model'):
                os.mkdir('./model')
            # torch.save(state, './model/' + str(epoch+1) + "_ckpt.pth')
            save_full_model(net, filename =  model_name)
            best_acc = test_acc
        print("==> Test loss:  ", round(test_loss, 4), " Test Acc: ", round(test_acc, 4), "===> Best Acc: ", round(best_acc, 4))
    ## draw loss and acc ##
    if isDraw:
        plot_loss(loss_list) #pytorch_utility.
        plot_acc(acc_list) #pytorch_utility.
    write_training_process(loss_list) #pytorch_utility.
    return
