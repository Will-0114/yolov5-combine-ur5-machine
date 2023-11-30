from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget,QDialog, QLabel, QTextEdit, QTextBrowser, QHBoxLayout, QVBoxLayout,QPushButton
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from PyQt5.QtCore import QTimer
import threading
import urx
import math
import socket
from View.UI_Main import * #連接畫面
from View.setvalue import * #設數值小視窗
from View.program1 import * #program輸入數值
import URuse
import URvis
import sys
from datetime import datetime
#import PySpin
import cv2
import warnings
import gc
import Gado
import numpy as np
from PIL import Image
import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# from PyQt5 import QtCore, QtGui, QtWidgets  ## switch to pyqt5
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLabel, QLineEdit, QPushButton, QMessageBox, QFileDialog
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QTextBrowser, QHBoxLayout, QVBoxLayout
#from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene  ## for qt5
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui import QIcon, QImage, QPixmap, QFont
#from Main_ui import *  ## created by qt designer 
#import win32com.client  ## 語音系統的lib
import time
import View
#import PG_CCD_Class_5 as pg  ## change to qt5
#from TPyArduino_Uno import *
#from Draw_Utility import draw_rectangle, write_text
from datetime import datetime
#import DIP_Class
from Tutils.tien_utility import findAllDirName, findAllDir, write_class
import os
from Tutils.DIP_Class import DIP
### resnet ###
#from ResNet.Keras_ResNet_Dog_Cat import ResNet, train
from Tutils import pytorch_utility, frcnn_utils
from Tutils.Draw_Utility import write_text
from Tutils.pytorch_utility import read_classes
from Tutils.yolov5_utils import yolov5_setup, yolov5_load_model, yolov5_predict
import torch
import Gado
#from Tutils_rs import rs_class
### pyinstaller problem ###
try:
    _fromUtf8 = PyQt5.QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = PyQt5.QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Pen_Params(object):
    color = 'blue'
    width = 2
    color_list = ['red', 'blue', 'yellow', 'black', 'white']

HOST = "192.168.0.3"   # The remote host
PORT = 30002              # The same port as used by the server

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

cmd = "set_digital_out(2,True)" + "\n" 
s.send (cmd.encode("utf-8"))
time.sleep(2)

error = 0.049464366411133956
print("Robot object is available as robot or rob")
robot_ur5 = URuse.UR_Control()
class iViLab_Main(QMainWindow):
    cvImage = None
    cvImg = None
    cvGoodImage = None
    cvDefectImage = None
    pen_params = Pen_Params()
    isPainting = False
    isSettingPosition = False
    roi_x = 100
    roi_y = 100
    roi_width = 100
    roi_height = 100
    dip = DIP()
    roi_image = None
    defect_x = 100 ## defect transfer position starting point
    defect_y = 100
    #resnet = ResNet()
    ## Signal ##
    FRCNN_train_signal = pyqtSignal()
    resnet_type_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    def __init__(self, parent = None):
        super(QMainWindow, self).__init__()
        self.ui = Ui_Main_Window()
        self.ui.setupUi(self)
        self.setWindowTitle('AI Applications-- iVi Lab, Dept. IE&M, Taipei Tech')
        self.setWindowIcon(QIcon('ivi_icon.ico')) 
        #call function
        self.grip() #爪子
        self.movelvalue() #線性數據

        self.movejvalue() #角度數據
        self.setvaluepage()
        self.movel() #線性移動
        self.MoveJoint() #六軸移動
        self.movetcp() #TCP方向
        self.updataData() #更新數據
        self.settohome() #回零
        self.URspeed() #速度
        self.stop() #停止機器
        self.freedrive() #自由操作
        self.sub = Subwindow()

        
        
        
        #self.largest_rect = QRect(0, 0, 400, 400)
        self.largest_rect = QRect(0, 0, self.ui.label_Image.width(), self.ui.label_Image.height())
        self.clip_rect = QRect(self.roi_x, self.roi_y, self.roi_width, self.roi_height)
        self.dragging = None
        self.det_list = None
        self.drag_offset = QPoint()
        self.linkEvent()      
        self.show()
        ### HSV Setup as black gray color ##
        self.hsv_low = [0, 0, 0]
        self.hsv_high = [179, 255, 255]
        self.load_params()
        self.mouse_offset_x = self.ui.label_Image.x() + self.ui.scrollArea.x()
        ## y shift 47 為 hardcode, not good, but do not know why yet
        self.mouse_offset_y = 47 #self.ui.label_Image.y() + self.ui.scrollArea.y()+ self.ui.menubar.y()
        #print("mouse offset: ", self.mouse_offset_x, self.mouse_offset_y)
        ## Load ResNet model list ##
        #self.resnet_model_path_list = findAllDir("./Model/ResNet_Model")
        #model_list = findAllDirName("./Model/ResNet_Model")
        #self.ui.comboBox_Model.addItems(model_list)
        ## Load FRCNNm odel list ## 
        #self.model_path_list = findAllDir("./Model/FRCNN_Model")
        #FRCNN_model_list = findAllDirName("./Model/FRCNN_Model")
        #self.ui.comboBox_Model_FRCNN.addItems(FRCNN_model_list)
        ## set up the resnet image size
        # self.resnet.size_image_x= 256  ## training using 256x256
        # self.resnet.size_image_y= 256
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")
        ## RealSense ##
        self.rs = None # 
        self.grabbed_color_image = None
        self.grabbed_depth_image = None
        self.ext = ".jpg"
        self.yolo_model = None
        self.conf_thres = 0.7
        self.iou_thres = 0.3
        self.isColorTracking = False
        self.bgr_list = list()
        self.hsv_list = list()
        self.com_list = list()
        self.xyz_list = [0,0,0]
        self.angle = 0
        self.range = None
        #combolist set
        combo = ['視角','機座','工具']
        self.ui.comboBox_list.addItems(combo)
        self.ui.comboBox_list.currentTextChanged.connect(self.movelvalue)
        self.ui.comboBox_list.currentTextChanged.connect(self.movel)
        self.ui.pushButton_3.clicked.connect(lambda : self.gettcp())
        


        return

    def linkEvent(self):
        # self.ui.pushButton_Load_Image.clicked.connect(lambda: self.load_image())
        # self.ui.spinBox_Scale.valueChanged.connect(lambda: self.change_image_scale())
        # self.ui.pushButton_Create_ROI.clicked.connect(lambda: self.create_roi())
        # self.ui.spinBox_ROI_Width.valueChanged.connect(lambda: self.change_roi_width())
        # self.ui.spinBox_ROI_Height.valueChanged.connect(lambda: self.change_roi_height())       
        # self.ui.pushButton_Save_Params.clicked.connect(lambda: self.save_params())
        # self.ui.pushButton_Save_Defect.clicked.connect(lambda: self.save_defect_image())
        #### transfer tabpage ####
        # self.ui.pushButton_Load_Image_Transfer.clicked.connect(lambda: self.load_good_image())
        # self.ui.pushButton_Load_Defect_Image.clicked.connect(lambda: self.load_defect_image())
        # self.ui.spinBox_Scale_Transfer.valueChanged.connect(lambda: self.change_good_image_scale())
        # self.ui.spinBox_Scale_Defect_Image.valueChanged.connect(lambda: self.change_defect_image_scale())
        # self.ui.pushButton_Set_HSV_Param.clicked.connect(lambda: self.set_hsv_by_image()) ## first time to set up the low high hsv params
        # self.ui.pushButton_Segment.clicked.connect(lambda: self.hsv_segment())
        # self.ui.pushButton_Set_Defect_Position.clicked.connect(lambda: self.set_defect_position())
        # self.ui.pushButton_Save_Patched_Image.clicked.connect(lambda: self.save_both_image())
        # self.ui.pushButton_Flip_Horizontal.clicked.connect(lambda:self.flip_defect(1))
        # self.ui.pushButton_Flip_Vertical.clicked.connect(lambda:self.flip_defect(0))
        # self.ui.pushButton_Blur_Defect.clicked.connect(lambda:self.blur_defect())
        # self.ui.pushButton_Sharp_Defect.clicked.connect(lambda:self.sharpen_defect())
        # self.ui.pushButton_Dark_Defect.clicked.connect(lambda: self.darken_defect())
        # self.ui.pushButton_Bright_Defect.clicked.connect(lambda: self.brighten_defect())
        ## tab chnage
        self.ui.tabWidget_AI.currentChanged.connect(lambda: self.onTabChange()) #changed!

        ### toolbutton
        self.ui.toolButton_Brighten.clicked.connect(lambda: self.brighten_base_image())
        self.ui.toolButton_Darken.clicked.connect(lambda: self.darken_base_image())
        self.ui.toolButton_Blur.clicked.connect(lambda: self.blur_base_image())
        self.ui.toolButton_Sharp.clicked.connect(lambda: self.sharpen_base_image())
        ### Menubar ###
        self.ui.actionLoad.triggered.connect(lambda: self.load_good_image()) #  load_image())
        self.ui.actionSave_Image.triggered.connect(lambda: self.save_good_image())
        ### YOLO Detection ###
        self.ui.pushButton_Load_YOLO_Model.clicked.connect(lambda: self.load_yolo_model())
        self.ui.pushButton_Load_Image_AI.clicked.connect(lambda: self.load_image_ai())
        self.ui.spinBox_Scale_AI.valueChanged.connect(lambda: self.change_good_image_scale_ai(spinner_index ='yolo'))
        self.ui.spinBox_Scale_AI_Detected.valueChanged.connect(lambda: self.change_yolo_image_scale_detect())
        self.ui.pushButton_Yolo_Detect.clicked.connect(lambda: self.yolo_detect())
        self.ui.spinBox_Yolo_Confidence.valueChanged.connect(lambda: self.change_confidence())
        self.ui.spinBox_Yolo_IOU.valueChanged.connect(lambda: self.change_iou())
        ##############################
        self.ui.toolButton_Flip_H.clicked.connect(lambda: self.flip_h_base_image())
        self.ui.toolButton_Flip_V.clicked.connect(lambda: self.flip_v_base_image())
        ## ResNet Training ##
        ## Hide some tabpage: self.setTabEnabled(tabIndex,True/False) #enable/disable the tab       
        self.ui.tabWidget_AI.removeTab(4)
        self.ui.tabWidget_AI.removeTab(5)
        self.ui.tabWidget_AI.removeTab(6)
        self.ui.tabWidget_AI.removeTab(7)
        self.ui.tabWidget_AI.removeTab(8)  
        #self.ui.tabWidget_AI.setTabEnabled(2, False)   
        #self.ui.tabWidget_AI.setTabEnabled(3, False)
        #self.ui.tabWidget_AI.setTabEnabled(4, False)
        #self.ui.tabWidget_AI.setTabEnabled(5, False)
        #self.ui.tabWidget_AI.setTabEnabled(6, False)
        ## PyTorch: FRCNN ##
        # self.ui.pushButton_Load_Image_FRCNN.clicked.connect(lambda: self.load_image_FRCNN())
        # self.ui.spinBox_Scale_FRCNN.valueChanged.connect(lambda: self.change_good_image_scale_FRCNN())
        # self.ui.pushButton_Select_Data_Folder_FRCNN.clicked.connect(lambda: self.open_training_data_folder())
        # self.ui.pushButton_Train_FRCNN.clicked.connect(lambda: self.train_FRCNN_ui())
        # self.ui.pushButton_Retrain_FRCNN.clicked.connect(lambda: self.retrain_FRCNN())
        # self.ui.pushButton_Load_FRCNN_Model.clicked.connect(lambda: self.load_FRCNN_model())
        # self.ui.pushButton_Classify_FRCNN.clicked.connect(lambda: self.FRCNN_classify())
        ##  Signal Connection ##
        #self.FRCNN_train_signal.connect(self.train_FRCNN)
        ###### RealSense related functions #####
        self.ui.toolButton_Stop_RS.setEnabled(False)
        self.ui.toolButton_Play_RS.setEnabled(False)
        self.ui.toolButton_Grab_RS.setEnabled(False)
        self.ui.toolButton_Start_RS.clicked.connect(lambda: self.start_rs())
        self.ui.toolButton_Stop_RS.clicked.connect(lambda: self.stop_rs())
        self.ui.toolButton_Grab_RS.clicked.connect(lambda: self.catch_all())
        self.ui.toolButton_Play_RS.clicked.connect(lambda: self.play_rs())
        self.ui.toolButton_Save_RS.clicked.connect(lambda: self.save_rs())
        self.timer = QTimer(self, interval = 5) ## in QTCore
        self.timer.timeout.connect(self.update_frame)
        ## Color Track ##
        self.ui.pushButton_Load_Image_Color.clicked.connect(lambda: self.load_image_ai())
        self.ui.spinBox_Scale_Color.valueChanged.connect(lambda: self.change_good_image_scale_ai(spinner_index = 'color'))
        self.ui.pushButton_Color_Track.clicked.connect(lambda: self.start_color_tracking())
        self.ui.pushButton_Color_Collect.clicked.connect(lambda: self.color_collect())
        self.ui.label_Image.mouseMoveEvent = self.color_track_mouse_move_event
        self.ui.label_Image.mousePressEvent = self.color_track_mouse_press_event
        ## set mouse event for label image
        self.ui.label_Image_Color.mouseMoveEvent = self.color_image_mouse_move_event
        self.ui.label_Image_Depth.mouseMoveEvent = self.depth_image_mouse_move_event
        self.ui.label_Image_IR.mouseMoveEvent = self.IR_image_mouse_move_event
        self.ui.toolButton_Det_RS.clicked.connect(lambda: self.det_data())
        self.ui.toolButton_Cali_RS.clicked.connect(lambda: self.catch_obj())
        return

    def start_color_tracking(self):
        if self.ui.pushButton_Color_Track.text() == "Start":
            self.isColorTracking = True
            self.ui.pushButton_Color_Track.setText("Stop")
            self.bgr_list.clear()
            self.hsv_list.clear()
        else:
            self.isColorTracking = False
            self.ui.pushButton_Color_Track.setText("Start")
            ## adding finding the max min of rgb, hsv_list
            (r_min, g_min, b_min), (r_max, g_max, b_max), (h_min, s_min, v_min), (h_max, s_max, v_max)= self.find_color_max_min(self.bgr_list, self.hsv_list)
            #print((r_min, g_min, b_min), (r_max, g_max, b_ax))
            self.ui.lineEdit_R_Min.setText(str(r_min))
            self.ui.lineEdit_R_Max.setText(str(r_max))
            self.ui.lineEdit_G_Min.setText(str(g_min))
            self.ui.lineEdit_G_Max.setText(str(g_max))
            self.ui.lineEdit_B_Min.setText(str(b_min))
            self.ui.lineEdit_B_Max.setText(str(b_max))
            ## hsv
            self.ui.lineEdit_H_Min.setText(str(h_min))
            self.ui.lineEdit_H_Max.setText(str(h_max))
            self.ui.lineEdit_S_Min.setText(str(s_min))
            self.ui.lineEdit_S_Max.setText(str(s_max))
            self.ui.lineEdit_V_Min.setText(str(v_min))
            self.ui.lineEdit_V_Max.setText(str(v_max))
        return

    def find_color_max_min(self, bgr_list, hsv_list):
        if len(bgr_list) == 0:
            r_min =0
            g_min =0
            b_min =0
            r_max =0
            g_max =0
            b_max =0
            h_min =0
            s_min =0
            v_min =0
            h_max =0
            s_max =0 
            v_max =0
            return (r_min, g_min, b_min), (r_max, g_max, b_max), (h_min, s_min, v_min), (h_max, s_max, v_max)
        import operator
        b_min =min(bgr_list, key=operator.itemgetter(0))[0]
        g_min =min(bgr_list, key=operator.itemgetter(1))[1]
        r_min =min(bgr_list, key=operator.itemgetter(2))[2]
        b_max =max(bgr_list, key=operator.itemgetter(0))[0]
        g_max =max(bgr_list, key=operator.itemgetter(1))[1]
        r_max =max(bgr_list, key=operator.itemgetter(2))[2]

        h_min =min(hsv_list, key=operator.itemgetter(0))[0]
        s_min =min(hsv_list, key=operator.itemgetter(1))[1]
        v_min =min(hsv_list, key=operator.itemgetter(2))[2]
        h_max =max(hsv_list, key=operator.itemgetter(0))[0]
        s_max =max(hsv_list, key=operator.itemgetter(1))[1]
        v_max =max(hsv_list, key=operator.itemgetter(2))[2]
        return (r_min, g_min, b_min), (r_max, g_max, b_max), (h_min, s_min, v_min), (h_max, s_max, v_max)

    def color_track_mouse_press_event(self, event):
        if self.cvGoodImage is not None:
            max_row, max_col, __ = self.cvGoodImage.shape
            pos_x = event.pos().x()
            pos_y = event.pos().y()
            if pos_x < max_col and pos_y < max_row and self.isColorTracking:
                b, g, r = self.cvGoodImage.item(pos_y, pos_x, 0), self.cvGoodImage.item(pos_y, pos_x, 1), self.cvGoodImage.item(pos_y, pos_x, 2) ## faster
                h, s, v = self.hsv.item(pos_y, pos_x, 0), self.hsv.item(pos_y, pos_x, 1), self.hsv.item(pos_y, pos_x, 2) ## faster
                self.bgr_list.append((b,g,r))
                self.hsv_list.append((h,s,v))
                msg = "(" +str(r) + "," + str(g) + ", " + str(b)+ ")" + " (" +str(h) + "," + str(s) + ", " + str(v)+ ")"
                self.ui.textEdit_Color.append(msg)
        return

    def color_track_mouse_move_event(self, event):
        if self.cvGoodImage is not None:
            max_row, max_col, __ = self.cvGoodImage.shape
            pos_x = event.pos().x()
            pos_y = event.pos().y()
            if pos_x < max_col and pos_y < max_row and self.isColorTracking:
                #b, g, r = self.grabbed_color_image.item(pos_y, pos_x, 0), self.grabbed_color_image.item(pos_y, pos_x, 1), self.grabbed_color_image.item(pos_y, pos_x, 2) ## faster
                b, g, r = self.cvGoodImage.item(pos_y, pos_x, 0), self.cvGoodImage.item(pos_y, pos_x, 1), self.cvGoodImage.item(pos_y, pos_x, 2) ## faster
                h, s, v = self.hsv.item(pos_y, pos_x, 0), self.hsv.item(pos_y, pos_x, 1), self.hsv.item(pos_y, pos_x, 2) ## faster
                self.ui.lineEdit_R.setText(str(r))
                self.ui.lineEdit_G.setText(str(g))
                self.ui.lineEdit_B.setText(str(b))
                self.ui.lineEdit_H.setText(str(h))
                self.ui.lineEdit_S.setText(str(s))
                self.ui.lineEdit_V.setText(str(v))
                msg = "(x, y) = (" + self.make_3_digit(str(pos_x)) + ", " + self.make_3_digit(str(pos_y))+ ") ==> (" + self.make_3_digit(str(r)) + "," + self.make_3_digit(str(g)) +"," + self.make_3_digit(str(b)) + ")"
                self.ui.statusbar.showMessage(msg)
                
        return

    def onTabChange(self):
        index = self.ui.tabWidget_AI.currentIndex()
        if index <2:  ## now only two tab maps
            self.ui.tabWidget_2.setCurrentIndex(index)
        if index ==2:
            self.ui.tabWidget_2.setCurrentIndex(1)
        return    
    ### mouse event ###
    def make_3_digit(self, string): ## make string exactly 3 digit
        if len(string) == 1:
            string = "  " + string
        elif len(string) ==2:
            string = " " + string
        return string

    def IR_image_mouse_move_event(self, event):
        if self.grabbed_ir_image is not None:
            max_row, max_col= self.grabbed_ir_image.shape  ## 1 channel gray image
            pos_x = event.pos().x()
            pos_y = event.pos().y()
            if pos_x < max_col and pos_y < max_row:
                gray = self.grabbed_ir_image.item(pos_y, pos_x) ## faster
                msg = "(x, y) = (" + self.make_3_digit(str(pos_x)) + ", " + self.make_3_digit(str(pos_y))+ ") ==> ( " + self.make_3_digit(str(gray)) + " )"
                #print(msg)
                self.ui.statusbar.showMessage(msg)

    def depth_image_mouse_move_event(self, event):
        ## depth is also three channel
        if self.grabbed_depth_image is not None:
            max_row, max_col, channel = self.grabbed_depth_image.shape
            pos_x = event.pos().x()
            pos_y = event.pos().y()
            if pos_x < max_col and pos_y < max_row:
                b, g, r = self.grabbed_depth_image.item(pos_y, pos_x, 0), self.grabbed_depth_image.item(pos_y, pos_x, 1), self.grabbed_depth_image.item(pos_y, pos_x, 2) ## faster
                depth = self.rs.get_depth(pos_x, pos_y)
                #depth = self.rs.g
                msg = "(x, y) = (" + self.make_3_digit(str(pos_x)) + ", " + self.make_3_digit(str(pos_y))+ ") ==> ("  \
                    + self.make_3_digit(str(r)) + "," + self.make_3_digit(str(g)) +"," + self.make_3_digit(str(b)) + ")" \
                    + "==> "  + str(round(depth, 3)) + " meter"    
                self.ui.statusbar.showMessage(msg)
        return

    def color_image_mouse_move_event(self, event):
        if self.grabbed_color_image is not None: # or self.cvGoodImage is not None:
            max_row, max_col, __ = self.grabbed_color_image.shape
            pos_x = event.pos().x()
            pos_y = event.pos().y()
            if pos_x < max_col and pos_y < max_row:
                b, g, r = self.grabbed_color_image.item(pos_y, pos_x, 0), self.grabbed_color_image.item(pos_y, pos_x, 1), self.grabbed_color_image.item(pos_y, pos_x, 2) ## faster
                msg = "(x, y) = (" + self.make_3_digit(str(pos_x)) + ", " + self.make_3_digit(str(pos_y))+ ") ==> (" + self.make_3_digit(str(r)) + "," + self.make_3_digit(str(g)) +"," + self.make_3_digit(str(b)) + ")"
                self.ui.statusbar.showMessage(msg)
            # if self.isColorTracking: ##Show color info on lineEdit_color
            #     b, g, r = self.cvGoodImage.item(pos_y, pos_x, 0), self.cvGoodImage.item(pos_y, pos_x, 1), self.cvGoodImage.item(pos_y, pos_x, 2) ## faster
            #     self.ui.lineEdit_R.setText(str(r))
            #     self.ui.lineEdit_G.setText(str(g))
            #     self.ui.lineEdit_B.setText(str(b))
        return

    def disable_buttons(self):
        self.ui.toolButton_Stop_RS.setEnabled(False)
        self.ui.toolButton_Play_RS.setEnabled(False)
        self.ui.toolButton_Grab_RS.setEnabled(False)
        self.ui.toolButton_Save_RS.setEnabled(False)
        return
    
    def enable_buttons(self):
        self.ui.toolButton_Stop_RS.setEnabled(True)
        self.ui.toolButton_Play_RS.setEnabled(True)
        self.ui.toolButton_Grab_RS.setEnabled(True)
        self.ui.toolButton_Save_RS.setEnabled(True)
        return

    @pyqtSlot()
    def save_rs(self):
        self.img_path = "./Save_Images"
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        fn_color = self.img_path+"/color_" + self.ui.lineEdit_Save_Prefix.text() +"_"+ dt_string + self.ext
        if self.grabbed_color_image is not None:
            cv2.imwrite(fn_color, self.grabbed_color_image)
        # fn_depth = self.img_path+"/depth_" + self.ui.lineEdit_Save_Prefix.text() +"_"+ dt_string + self.ext
        # if self.grabbed_depth_image is not None:
        #     cv2.imwrite(fn_depth, self.grabbed_depth_image)
        # fn_ir = self.img_path+"/ir_" + self.ui.lineEdit_Save_Prefix.text() +"_"+ dt_string + self.ext
        # if self.grabbed_ir_image is not None:
        #     cv2.imwrite(fn_ir, self.grabbed_ir_image)
        return

    @pyqtSlot()
    def grab_rs(self):
        if not self.rs.isActive:
            self.ui.statusbar.showMessage("Please start Cameras first...")
            return
        self.grabbed_color_image = self.rs.get_color_image()
        self.grabbed_depth_image = self.rs.get_depth_image()
        
        np.set_printoptions(threshold=10000)
        print(self.grabbed_depth_image)

        self.grabbed_ir_image = self.rs.get_ir_image()
        self.displayRGBImage(self.grabbed_color_image, True) ## default True
        self.displayDepthImage(self.grabbed_depth_image, True)
        self.displayIRImage(self.grabbed_ir_image, True)
        return

    @pyqtSlot()
    def play_rs(self):
        if self.ui.toolButton_Play_RS.text() == "Play":
            self.timer.start()
            self.ui.toolButton_Play_RS.setText("Stop")
            self.ui.toolButton_Play_RS.setStyleSheet("color : red")
            #self.ui.toolButton_Play.setStyleSheet('QtoolButton {color: red;}')
            self.ui.toolButton_Play_RS.setFont(QFont('Arial', 11))
        else:
            self.timer.stop()
            self.ui.toolButton_Play_RS.setText("Play")
            self.ui.toolButton_Play_RS.setStyleSheet('QtoolButton {color: black;}')
        return

    @pyqtSlot()
    def update_frame(self):
        self.grabbed_color_image = self.rs.get_color_image()
        self.grabbed_depth_image = self.rs.get_depth_image()

        self.grabbed_ir_image = self.rs.get_ir_image()
        self.displayRGBImage(self.grabbed_color_image, True) ## default True
        self.displayDepthImage(self.grabbed_depth_image, True)
        self.displayIRImage(self.grabbed_ir_image, True)
        return

    def displayRGBImage(self, img, window=True):
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.ui.label_Image_Color.setPixmap(QPixmap.fromImage(outImage))
    
    def displayIRImage(self, img, window=True): ## IR is gray image
        if len(img.shape)==2:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
            
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.ui.label_Image_IR.setPixmap(QPixmap.fromImage(outImage))

    def det_data(self):
        text = self.ui.com_list.currentText()
        for i in range(len(self.det_list)):
            if text == str(int(self.det_list[i][5])):
                self.xyz_list[0] = (self.det_list[i][0] + self.det_list[i][2]) /2
                self.xyz_list[1] = (self.det_list[i][1] + self.det_list[i][3]) /2
                x1 = self.det_list[i][0]
                x2 = self.det_list[i][2]
                y1 = self.det_list[i][1]
                y2 = self.det_list[i][3]
                self.cal_angle(int(x1),int(y1),int(x2),int(y2))
                print(self.angle)
        
        depth = self.rs.get_depth(int(self.xyz_list[0]), int(self.xyz_list[1]))
        self.xyz_list[2] = depth
        print(self.xyz_list)
        self.read_xyz_value()
        return

    def cal_angle(self,x1,y1,x2,y2):
        img = cv2.imread(self.good_filename, cv2.IMREAD_COLOR)
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
            #print(rect)
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
                self.angle =rect[2]
                return 
                #for n in range(len(yolov5_predict.det_list)):
                #    lst = yolov5_predict.det_list[n]
                #    if (cX, cY)  >= (lst[0], lst[1]) and (cX, cY)  <= (lst[2], lst[3]):
                #        cv2.drawContours(img, [box], -1,(0, 0, 255), 2)
                #        cv2.imshow("img", img)
                #        cv2.waitKey(-1)
    

    def displayDepthImage(self, img, window=True):
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.ui.label_Image_Depth.setPixmap(QPixmap.fromImage(outImage))

    @pyqtSlot()
    def start_rs(self):
        from Tutils_rs import rs_class
        # try:
                
        self.rs = rs_class.RealSenseL515()
        
        self.rs.stop()
        self.rs.start_rs()
            # print('789')
            # self.rs.display_cv()
            
        self.ui.statusbar.showMessage("Creating realSense Cameras...")
        self.ui.toolButton_Start_RS.setEnabled(False)
        self.enable_buttons()
        # except:
        #     QMessageBox.warning(self, "Warning", "Please connect RealSense properly...")
        #     self.rs = None
        return

    @pyqtSlot()
    def stop_rs(self):
        self.timer.stop()
        if self.rs is None:
            self.ui.statusbar.showMessage("RealSense Cameras are not created")
            return
        self.rs.stop()  ## make sure stop playing
        self.rs.quit_rs()
        self.ui.statusbar.showMessage("RealSense Cameras are closed")
        self.ui.toolButton_Start_RS.setEnabled(True)
        self.disable_buttons()
        return

    def train_FRCNN_ui(self):
        self.FRCNN_train_signal.emit()
        return

    def FRCNN_classify(self):
        self.class_list = ["background", "c1", "c2", "c3"]
        cvImage = self.cvGoodImg
        #cvImage = cv2.imread(filename, 1)
        self.FRCNN_threshold = self.ui.spinBox_Threshold_FRCNN.value()/100
        boxes, pred_cls, pred_score, img = frcnn_utils.object_detection_frcnn_cvImg(self.FRCNN_model, cvImage, class_list = self.class_list, threshold=self.FRCNN_threshold, isShow =False)
        ##print("BBoxes/Labels/Prob.: ", boxes, pred_cls, pred_score)
        self.display_img(img, self.ui.label_Image)
        return

    def load_image_FRCNN(self):
        self.filename = QFileDialog.getOpenFileName(self, 'Open FRCNN Testing Image File', './')
        if self.filename =="":
            return None
        self.good_filename = self.filename[0]
        print("Open Good image file: ", self.good_filename)
        self.range = cv2.imread(self.good_filename, cv2.IMREAD_COLOR)
        self.cvGoodImage = cv2.imread(self.good_filename)
        if self.cvGoodImage is None:
            #print("No image is loaded")
            self.ui.label_Image.setText("No image is loaded")
            return
        self.scale_Good = self.ui.spinBox_Scale_FRCNN.value() / 100
        self.cvGoodImg = cv2.resize(self.cvGoodImage, (0, 0), fx= self.scale_Good, fy=self.scale_Good)  
        self.display_img(self.cvGoodImg, self.ui.label_Image)
        return

    def change_good_image_scale_FRCNN(self):
        self.scale_good = self.ui.spinBox_Scale_FRCNN.value() / 100
        if self.cvGoodImage is None:
            self.ui.label_Image.setText("Please load an image first")
            return
        self.cvGoodImg = cv2.resize(self.cvGoodImage, (0, 0), fx= self.scale_good, fy=self.scale_good)
        self.display_img_on_label(self.cvGoodImg)
        #self.largest_rect = QRect(0, 0, self.ui.label_Image.width(), self.ui.label_Image.height())
        #print(self.ui.label_Image.width(), self.ui.label_Image.height())
        #print("shape:", self.cvImg.shape)
        return

    def load_FRCNN_model(self):
        index = self.ui.comboBox_Model_FRCNN.currentIndex()
        model_dir = self.model_path_list[index]
        dialog = QFileDialog()
        filename = dialog.getOpenFileName(self, 'Select model file', model_dir)
        #self.modir_dir = dialog.getOpenFileName(self, 'Select an awesome directory')  #dialog.getExistingDirectory(self, 'Select an awesome directory', model_dir)
        if filename[0] =="":
            return None
        self.fn = filename[0]
        ## load FRCNN model
        t_start = time.clock()
        self.FRCNN_model = frcnn_utils.load_full_model(filename=self.fn)
        self.ui.statusbar.showMessage("FRCNN Model is loaded:"+ self.fn)
        ## always test an image
        #model = resnet.load_model(path="./Model/ResNet_Model/Cat&Dog", fn = "ResNet34__model.h5")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")
        self.FRCNN_model = self.FRCNN_model.to(self.device)
        
        self.class_list = frcnn_utils.read_classes(classes_path = model_dir+"\\classes.txt")
        self.class_list.insert(0, "background")
        #self.class_list = ["background", "c1", "c2", "c3"]
        filename =  "./test_image.jpg"
        cvImage = cv2.imread("./FRCNN_test.bmp", -1)
        #cvImage = cv2.imread(filename, 1)
        self.FRCNN_threshold = self.ui.spinBox_Threshold_FRCNN.value()/100
        boxes, pred_cls, pred_score, img = frcnn_utils.object_detection_frcnn_cvImg(self.FRCNN_model, cvImage, class_list = self.class_list, threshold=self.FRCNN_threshold, isShow =False)
        ## print("BBoxes/Labels/Prob.: ", boxes, pred_cls, pred_score)
        t_end = time.clock()
        print("Loading model spent: ", round((t_end - t_start), 3), " sec.")
        ## show messasge box ###
        buttonReply = QMessageBox.question(self, 'iVi Lab Message', "FRCNN Model is loaded", QMessageBox.Ok) # QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return

    def retrain_FRCNN(self):
        self.num_classes = self.ui.spinBox_Num_Classes_FRCNN.value()
        self.num_epoch = self.ui.spinBox_No_Epoch_FRCNN.value()
        self.batch_size = self.ui.spinBox_Batch_Size_FRCNN.value()
        self.learnig_rate = float(self.ui.lineEdit_Learning_Rate_FRCNN.text())
        self.evaluation_period = self.ui.spinBox_Evaluation_Period_FRCNN.value()
        self.model_name = self.ui.lineEdit_Model_Name_FRCNN.text()
        if self.model_name == "":
            QMessageBox.warning(self, "Warning", "Please input model name")
            return
        model_dir = "./Model/FRCNN"
        dialog = QFileDialog()
        filename = dialog.getOpenFileName(self, 'Select model file', model_dir)
        #self.modir_dir = dialog.getOpenFileName(self, 'Select an awesome directory')  #dialog.getExistingDirectory(self, 'Select an awesome directory', model_dir)
        if filename[0] =="":
            return None
        self.fn = filename[0]
        # self.model_path = "./model/FRCNN_Model/"+ self.model_name
        # if not os.path.isdir(self.model_path):
        #     os.mkdir(self.model_path)
        self.model_name =  self.fn #self.model_path + "/frcnn_" + self.model_name +".pth" #self.model_name + "/"+
        self.ui.statusbar.showMessage("Set FRCNN model name: " + self.model_name)
        self.train_data = self.training_dir+ "/train_data"
        self.test_data = self.training_dir+ "/test_data"
        frcnn_utils.retrain_frcnn(no_epoch = self.num_epoch, batch_size = self.batch_size, train_data_path =self.train_data, 
             test_data_path = self.test_data, load_model_name = self.model_name, save_model_name =self.model_name, evaluate_period = self.evaluation_period )
        return

    def train_FRCNN(self):  
        self.num_classes = self.ui.spinBox_Num_Classes_FRCNN.value()
        self.num_epoch = self.ui.spinBox_No_Epoch_FRCNN.value()
        self.batch_size = self.ui.spinBox_Batch_Size_FRCNN.value()
        self.learnig_rate = float(self.ui.lineEdit_Learning_Rate_FRCNN.text())
        self.evaluation_period = self.ui.spinBox_Evaluation_Period_FRCNN.value()
        self.model_name = self.ui.lineEdit_Model_Name_FRCNN.text()
        if self.model_name == "":
            QMessageBox.warning(self, "Warning", "Please input model name")
            return
        self.model_path = "./model/FRCNN_Model/"+ self.model_name
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        self.model_name =  self.model_path + "/frcnn_" + self.model_name +".pth" #self.model_name + "/"+
        self.ui.statusbar.showMessage("Load FRCNN model name: " + self.model_name)
        self.train_data = self.training_dir+ "/train_data"
        self.test_data = self.training_dir+ "/test_data"
        self.classes_list = frcnn_utils.read_classes(classes_path = self.train_data + "/classes.txt")
        frcnn_utils.write_clases(self.classes_list, path =self.model_path)
        frcnn_utils.train_frcnn(no_epoch = self.num_epoch, num_classes = self.num_classes, batch_size = self.batch_size, train_data_path =self.train_data, 
             test_data_path = self.test_data, save_model_name =self.model_name, evaluate_period = self.evaluation_period )
        return

    def train_resnet(self):
        ##1. read dir, and create class.text
        #self.class_list = findAllDirName(self.training_dir)
        #print(self.class_list)
        #write_class(self.class_list, self.training_dir)
        ##2. Load param

        self.image_row = int(self.ui.lineEdit_Image_Size_Row.text())
        self.image_col = int(self.ui.lineEdit_Image_Size_Col.text())
        self.no_epoch = self.ui.spinBox_No_Epoch.value()
        self.batch_size = self.ui.spinBox_Batch_Size.value()
        self.resnet_type = self.ui.comboBox_ResNetType.currentText()
        self.model_name = self.ui.lineEdit_Model_Name.text()
        self.isHFlip = self.ui.checkBox_Flip_Horizontal.isChecked()
        self.isVFlip = self.ui.checkBox_Flip_Vertical.isChecked()
        self.training_dir_augment = self.training_dir + "_augment"
        self.val_dir = self.training_dir + "/test"
        self.train_dir = self.training_dir + "/train"
        self.lr = float(self.ui.lineEdit_Learning_Rate.text())
        ## model_name 用來create a dir in model
        if self.model_name == "":
            QMessageBox.warning(self, "Warning", "Please input model name")
            return
        self.model_path = "./model/ResNet_Model/"+ self.model_name

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        self.model_name =  self.model_path + "/" + self.resnet_type + ".pth" #self.model_name + "/"+
        self.ui.statusbar.showMessage("Set ResNet model name: " + self.model_name)
        ## start training ##
        # train(self.image_row, self.image_col, self.no_epoch, self.batch_size,
        #         self.training_dir, self.training_dir_augment, self.resnet_type, self.model_path, self.model_name, 
        #         isDataAugment = True, isHFlip = self.isHFlip, isVFlip = self.isVFlip)
        ## Change to PyTorch: 
        pytorch_utility.train_ResNet(train_image_dir = self.train_dir, val_image_dir = self.val_dir, no_epachs = self.no_epoch, lr = 0.01,
                     type = self.resnet_type, pretrained = True, batch_size = self.batch_size, model_name = self.model_name, isDraw = False)
        return

    def open_training_data_folder(self):
        #data_dir = QFileDialog.getExistingDirectory(self, 'Select a folder:', './', QtGui.QFileDialog.ShowDirsOnly)
        dialog = QFileDialog()
        self.training_dir = dialog.getExistingDirectory(self, 'Select an awesome directory', './Training_Data')
        print(self.training_dir)
        if self.training_dir  =="":
            return
        self.ui.lineEdit_Data_Dir.setText(self.training_dir)
        self.ui.lineEdit_Data_Dir_FRCNN.setText(self.training_dir)
        return

    def yolo_detect(self):
        if self.yolo_model is None:
            buttonReply = QMessageBox.question(self, 'iVi Lab Message', "Please load Yolo Model first", QMessageBox.Ok) # QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return
        t_start = time.clock()
        self.display_img, self.det_list = yolov5_predict(self.yolo_model, self.cvGoodImage, self.device, names=self.class_names, half=self.half, img_size = self.img_size, conf_thres = self.conf_thres, iou_thres = self.iou_thres, view_img= False, isRandomColor=False)
        t_end = time.clock()
        self.display_img_on_label(self.display_img)
        self.ui.statusbar.showMessage("Prediction spends: " + str(round((t_end - t_start), 3)) + " sec.")
        for i in range(len(self.det_list)):
            self.com_list.append(str(int(self.det_list[i][5])))
        self.ui.com_list.addItems(self.com_list)
        
        
        return 

    def load_image_ai(self):
        self.filename = QFileDialog.getOpenFileName(self, 'Open Testing Image File', './')
        if self.filename =="":
            return None
        self.good_filename = self.filename[0]
        #print("Open Good image file: ", self.good_filename)
        self.ui.statusbar.showMessage("Open Good image file: " + self.good_filename)
        self.cvGoodImage = cv2.imdecode(np.fromfile(self.good_filename,dtype=np.uint8),-1) #v2.imread(self.good_filename)
        if self.cvGoodImage is None:
            #print("No image is loaded")
            self.ui.label_Image.setText("No image is loaded")
            return
        self.hsv = cv2.cvtColor(self.cvGoodImage, cv2.COLOR_BGR2HSV)
        self.scale_Good = self.ui.spinBox_Scale_AI.value() / 100
        self.cvGoodImg = cv2.resize(self.cvGoodImage, (0, 0), fx= self.scale_Good, fy=self.scale_Good)  
        self.display_img(self.cvGoodImg, self.ui.label_Image)
        return 

    def change_good_image_scale_ai(self, spinner_index="yolo"):
        ## several spinner scale share this function by giving spinner_index
        if spinner_index == 'yolo':
            self.scale_good = self.ui.spinBox_Scale_AI.value() / 100
        elif spinner_index == 'color':
            self.scale_good = self.ui.spinBox_Scale_Color.value() / 100
        else:
            self.scale_good = self.ui.spinBox_Scale_AI.value() / 100
        if self.cvGoodImage is None:
            self.ui.label_Image.setText("Please load an image first")
            return
        self.cvGoodImg = cv2.resize(self.cvGoodImage, (0, 0), fx= self.scale_good, fy=self.scale_good)
        self.display_img_on_label(self.cvGoodImg)
        #self.largest_rect = QRect(0, 0, self.ui.label_Image.width(), self.ui.label_Image.height())
        #print(self.ui.label_Image.width(), self.ui.label_Image.height())
        #print("shape:", self.cvImg.shape)
        return

    def change_yolo_image_scale_detect(self):
        self.scale_detect = self.ui.spinBox_Scale_AI_Detected.value() / 100
        if self.display_img is None:
            self.ui.label_Image.setText("Please detect first")
            return
        self.display_img1 = cv2.resize(self.display_img, (0, 0), fx= self.scale_detect, fy=self.scale_detect)
        self.display_img_on_label(self.display_img1)
        #self.largest_rect = QRect(0, 0, self.ui.label_Image.width(), self.ui.label_Image.height())
        #print(self.ui.label_Image.width(), self.ui.label_Image.height())
        #print("shape:", self.cvImg.shape)
        return

    def change_confidence(self):
        self.conf_thres = self.ui.spinBox_Yolo_Confidence.value() / 100
        return
    
    def change_iou(self):
        self.iou_thres = self.ui.spinBox_Yolo_IOU.value() / 100
        return

    def load_yolo_model(self):
        ## get current index of combobox model
        #index = self.ui.comboBox_Model.currentIndex()
        yolo_model_dir = "./weights"#self.resnet_model_path_list[index]
        dialog = QFileDialog()
        filename = dialog.getOpenFileName(self, 'Select model file', yolo_model_dir)
        #self.modir_dir = dialog.getOpenFileName(self, 'Select an awesome directory')  #dialog.getExistingDirectory(self, 'Select an awesome directory', model_dir)
        if filename[0] =="":
            return None
        self.fn = filename[0]
        #self.resnet_model=self.resnet.load_model(path =self.model_path_list[index], fn =filename )
        ## load pytorch model
        t_start = time.clock()
        self.img_size = 800
        self.yolo_model, self.device, self.half, self.class_names, colors = yolov5_load_model(model_path= self.fn, imgsz = self.img_size)
        #self.resnet_model = pytorch_utility.load_full_model(filename=self.fn)
        t_end = time.clock()
        print("Loading model spent: ", round((t_end - t_start), 3), " sec.")
        self.ui.statusbar.showMessage("YOLOv5 Model is loaded:"+ self.fn + "   timespan: "+ str(round((t_end - t_start), 3)) + " sec.")
        ## always test an image
        #model = resnet.load_model(path="./Model/ResNet_Model/Cat&Dog", fn = "ResNet34__model.h5")
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # else: 
        #     self.device = torch.device("cpu")
        filename =  "./test.jpg" ## testing image only
        img0 = cv2.imread(filename, -1)
        self.conf_thres = 0.7
        self.iou_thres = 0.3
        self.img_size = 800
        display_img, det_list = yolov5_predict(self.yolo_model, img0, self.device, names=self.class_names, half=self.half, img_size = self.img_size, conf_thres = self.conf_thres, iou_thres = self.iou_thres, view_img= False, isRandomColor=False)
        ## for testing only, save time when running program
        self.ui.textEdit_Info.append("Model loading is done ..." + "\n")
        ## show messasge box ###
        buttonReply = QMessageBox.question(self, 'iVi Lab Message', "AI Model is loaded", QMessageBox.Ok) # QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        #self.display_img(display_img, self.ui.label_Image) #display_img_on_label(display_img)
        return

    def save_good_image(self): ## save image without asking name, so save to its original filename
        (row, col, channel)= self.cvGoodImage.shape
        self.cvGoodImage = cv2.resize(self.cvGoodImg, (row, col))
        ## over-written the original image
        cv2.imwrite(self.good_filename , self.cvGoodImage)
        self.ui.statusbar.showMessage("Save image: " + self.good_filename)
        return

    def brighten_defect(self):
        self.cvDefectImg = self.dip.change_brightness(self.cvDefectImg, value = 5)
        self.display_img(self.cvDefectImg, self.ui.label_Defect_Image)
        return

    def darken_defect(self):
        self.cvDefectImg = self.dip.change_brightness(self.cvDefectImg, value = -5)
        self.display_img(self.cvDefectImg, self.ui.label_Defect_Image)
        return

    def sharpen_defect(self):
        self.cvDefectImg = self.dip.sharpen(self.cvDefectImg)
        self.display_img(self.cvDefectImg, self.ui.label_Defect_Image)
        return

    def blur_defect(self):
        self.cvDefectImg = self.dip.blur(self.cvDefectImg)
        self.display_img(self.cvDefectImg, self.ui.label_Defect_Image)
        return

    def sharpen_base_image(self):
        self.cvGoodImg = self.dip.sharpen(self.cvGoodImg)
        self.display_img(self.cvGoodImg, self.ui.label_Image)
        return

    def flip_h_base_image(self):
        self.cvGoodImg = self.dip.flip(self.cvGoodImg, 1)
        self.display_img(self.cvGoodImg, self.ui.label_Image)
        return
    
    def flip_v_base_image(self):
        self.cvGoodImg = self.dip.flip(self.cvGoodImg, 0)
        self.display_img(self.cvGoodImg, self.ui.label_Image)
        return

    def blur_base_image(self):
        self.cvGoodImg = self.dip.blur(self.cvGoodImg)
        self.display_img(self.cvGoodImg, self.ui.label_Image)
        return

    def flip_defect(self, code):
        self.cvDefectImg = self.dip.flip(self.cvDefectImg, code)
        self.display_img(self.cvDefectImg, self.ui.label_Defect_Image)
        return

    def darken_base_image(self):
        self.cvGoodImg = self.dip.change_brightness(self.cvGoodImg, value = -5)
        self.display_img(self.cvGoodImg, self.ui.label_Image)
        return
    def brighten_base_image(self):
        ##print("Brighten the good image")
        self.cvGoodImg = self.dip.change_brightness(self.cvGoodImg, value = 5)
        self.display_img(self.cvGoodImg, self.ui.label_Image)
        return

    def save_both_image(self):
        if self.ui.pushButton_Set_Defect_Position.text()=='Patch':
            QMessageBox.warning(self, "Warning", "Please patch image first")
            return
        now = datetime.now()
        time_tag = now.strftime("%Y_%m_%d_%H_%M_%S")
        #fn = defect_name+"_"+time_tag+".bmp"
        print(self.good_filename)
        base = os.path.basename(self.good_filename)
        basefilename = base.split(".")
        filename_in = "./In_Images/" + basefilename[0] + "_" + time_tag + ".bmp"
        filename_out = "./Out_Images/" + basefilename[0] + "_" + time_tag + ".bmp"
        ## resize back to original size
        (row, col, channel) = self.cvGoodImage.shape
        patched_defect_image = cv2.resize(self.patched_defect_image, (row, col))
        ##cv2.imshow("final image", patched_defect_image)
        ##cv2.waitKey(0)
        cv2.imwrite(filename_in, patched_defect_image)
        cv2.imwrite(filename_out, self.cvGoodImage) ## original good image
        return

    def save_defect_image(self):
        if self.ui.pushButton_Create_ROI.text()=='Crop':
            #print("No roi_image crop yet, please press crop button")
            QMessageBox.warning(self, "Warning", "Please click 'Crop' first")
            return
        if self.roi_image is None:
            print("No roi_image crop yet, please press crop button")
            return
        now = datetime.now()
        time_tag = now.strftime("%Y_%m_%d_%H_%M_%S")
        #fn = defect_name+"_"+time_tag+".bmp"
        filename = "./SB_Defect_Bank/" + self.ui.lineEdit_Image_Name.text() + time_tag + ".bmp"
        cv2.imwrite(filename, self.roi_image)
        return

    def set_defect_position(self):
        if self.ui.pushButton_Set_Defect_Position.text()=='Set Defect Position':
            self.isSettingPosition = True
            ## 必須加入這行，才會將cvGoodImage load 進來，否則會是最近使用load 的影像
            self.pixMap, w, h= self.convert_to_pixmap(self.cvGoodImg) # rescaled image
            self.ui.pushButton_Set_Defect_Position.setText("Patch")
        else:
            self.ui.pushButton_Set_Defect_Position.setText("Set Defect Position")
            #print("Done") ## draw defect image on the good one
            self.isSettingPosition = False
            ## defect.x and defect.y are the position of mouse to set the defect position
            ## cvGoodImg: scaled good image, defect_image: hsv_segmented image
            self.patched_defect_image = self.dip.patch(self.cvGoodImg, self.defect_image, x=self.defect_x, y=self.defect_y )
            self.display_img(self.patched_defect_image, self.ui.label_Image)
        return

    def change_roi_width(self):
        self.roi_width = self.ui.spinBox_ROI_Width.value()
        return

    def change_roi_height(self):
        self.roi_height = self.ui.spinBox_ROI_Height.value()
        return

    def create_roi(self):
        if self.ui.pushButton_Create_ROI.text()=='Create ROI':
            self.isPainting = True
            self.ui.pushButton_Create_ROI.setText('Crop')
            self.pixMap, w, h= self.convert_to_pixmap(self.cvImg) # rescaled image
            self.ui.spinBox_ROI_Width.setValue(100)
            self.ui.spinBox_ROI_Height.setValue(100)
        else: 
            self.isPainting = False
            self.ui.pushButton_Create_ROI.setText('Create ROI')
            ## draw roi here or release mouse there
            x_start = self.clip_rect.x()
            y_start = self.clip_rect.y()
            x_end = x_start + self.clip_rect.width()
            y_end = y_start + self.clip_rect.height()
            ## self.cvImg: is resized image
            self.roi_image = self.dip.crop_Image(self.cvImg, x_start, y_start, x_end, y_end, False)
            self.display_img(self.roi_image, self.ui.label_ROI_Image)
            ## self.roi_image 可能會太大，記得要scale back to original size
        return

    def change_defect_image_scale(self):
        self.scale_defect = self.ui.spinBox_Scale_Defect_Image.value() / 100
        if self.cvDefectImage is None:
            self.ui.label_Image.setText("Please load an image first")
            return
        self.cvDefectImg = cv2.resize(self.cvDefectImage, (0, 0), fx= self.scale_defect, fy=self.scale_defect)
        self.display_img(self.cvDefectImg, self.ui.label_Defect_Image)
        return

    def change_good_image_scale(self):
        self.scale_good = self.ui.spinBox_Scale_Transfer.value() / 100
        if self.cvGoodImage is None:
            self.ui.label_Image.setText("Please load an image first")
            return
        self.cvGoodImg = cv2.resize(self.cvGoodImage, (0, 0), fx= self.scale_good, fy=self.scale_good)
        self.display_img_on_label(self.cvGoodImg)
        #self.largest_rect = QRect(0, 0, self.ui.label_Image.width(), self.ui.label_Image.height())
        #print(self.ui.label_Image.width(), self.ui.label_Image.height())
        #print("shape:", self.cvImg.shape)
        return

    def change_image_scale(self):
        self.scale = self.ui.spinBox_Scale.value() / 100
        if self.cvImage is None:
            self.ui.label_Image.setText("Please load an image first")
            return
        self.cvImg = cv2.resize(self.cvImage, (0, 0), fx= self.scale, fy=self.scale)
        self.display_img_on_label(self.cvImg)
        self.largest_rect = QRect(0, 0, self.ui.label_Image.width(), self.ui.label_Image.height())
        #print(self.ui.label_Image.width(), self.ui.label_Image.height())
        #print("shape:", self.cvImg.shape)
        return

    def load_image(self):
        filename = QFileDialog.getOpenFileName(self, 'Open Image File', './')
        if filename[0] =="":
            return None
        self.fn = filename[0]
        #print("Open image file: ", filename)
        self.cvImage = cv2.imread(filename[0])
        self.scale = self.ui.spinBox_Scale.value() / 100
        self.cvImg = cv2.resize(self.cvImage, (0, 0), fx= self.scale, fy=self.scale)
        #print(type(self.cvImage))
        if self.cvImage is None:
            #print("No image is loaded")
            self.ui.label_Image.setText("No image is loaded")
            return
        self.display_img_on_label(self.cvImg)
        self.isSegmented = False  ## 用來確認user 必須要做segment
        return
    
    def load_good_image(self):
        self.filename = QFileDialog.getOpenFileName(self, 'Open Good Image File', './')
        if self.filename[0] =="":
            return None
        self.good_filename = self.filename[0]
        print("Open Good image file: ", self.good_filename)
        self.cvGoodImage = cv2.imread(self.good_filename)
        if self.cvGoodImage is None:
            #print("No image is loaded")
            self.ui.label_Image.setText("No image is loaded")
            self.hsv = cv2.cvtColor(self.cvGoodImage.shape, cv2.COLOR_BGR2HSV)
            return
        self.scale_Good = self.ui.spinBox_Scale_Transfer.value() / 100
        self.cvGoodImg = cv2.resize(self.cvGoodImage, (0, 0), fx= self.scale_Good, fy=self.scale_Good)  
        self.display_img(self.cvGoodImg, self.ui.label_Image)
        return
    
    def load_defect_image(self):
        filename = QFileDialog.getOpenFileName(self, 'Open Defect Image File', './SB_Defect_Bank')
        if filename[0] =="":
            return None
        #print("Open image file: ", filename)
        self.cvDefectImage = cv2.imread(filename[0])
        if self.cvDefectImage is None:
            #print("No image is loaded")
            self.ui.label_Defect_Image.setText("No image is loaded")
            return
        self.scale_defect = self.ui.spinBox_Scale_Defect_Image.value() / 100
        self.cvDefectImg = cv2.resize(self.cvDefectImage, (0, 0), fx= self.scale_defect, fy=self.scale_defect)
        #print(type(self.cvImage))
       
        self.display_img(self.cvDefectImg, self.ui.label_Defect_Image)
        return 

    def display_img_on_label(self, cvImage):
        self.pixMap, w, h= self.convert_to_pixmap(cvImage)
        self.pixMap.scaled(w, h, PyQt5.QtCore.Qt.KeepAspectRatio)
        self.ui.label_Image.setPixmap(self.pixMap)
        self.ui.label_Image.setAlignment(PyQt5.QtCore.Qt.AlignTop) 
        self.ui.label_Image.setScaledContents(False)
        #self.label_Image.setMinimumSize(1,1)
        self.ui.label_Image.setFixedWidth(w)
        self.ui.label_Image.setFixedHeight(h)
        self.ui.label_Image.show()
        return
    
    def display_img(self, cvImage, label):
        self.pixMap, w, h= self.convert_to_pixmap(cvImage)
        self.pixMap.scaled(w, h, PyQt5.QtCore.Qt.KeepAspectRatio)
        label.setPixmap(self.pixMap)
        label.setAlignment(PyQt5.QtCore.Qt.AlignTop) 
        label.setScaledContents(False)
        #self.label_Image.setMinimumSize(1,1)
        label.setFixedWidth(w)
        label.setFixedHeight(h)
        label.show()
        return
    
    def convert_to_pixmap(self, cvImg):
        img = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB) ## 改為 RGB: Pil_image
        height, width, byteValue = cvImg.shape
        byteValue = byteValue * width
        ## convert to pixmap
        mQImage = PyQt5.QtGui.QImage(img, width, height, byteValue, PyQt5.QtGui.QImage.Format_RGB888)
        pixMap = PyQt5.QtGui.QPixmap.fromImage(mQImage)
        return pixMap, width, height

    ###  mouse event for painter ###
    def paintEvent(self, event):      
        if self.isPainting:
            painter = PyQt5.QtGui.QPainter()
            painter.begin(self.ui.label_Image.pixmap())
            # pen = QtGui.QPen()
            # pen.setWidth(self.pen_params.width)
            # pen.setColor(QtGui.QColor(self.pen_params.color))
            #painter.setPen(pen)
            #painter.fillRect(event.rect(), QBrush(Qt.white))
            painter.setRenderHint(PyQt5.QtGui.QPainter.Antialiasing)
            painter.setPen(PyQt5.QtGui.QPen(PyQt5.QtGui.QBrush(PyQt5.Qt.red), 1, PyQt5.Qt.DashLine))
            #painter.drawRect(self.largest_rect)      
            #painter.setPen(QPen(Qt.black))  ## color
            ## draw image
            self.largest_rect = PyQt5.QtCore.QRect(0, 0, self.ui.label_Image.width(), self.ui.label_Image.height())
            painter.drawPixmap(0, 0, self.pixMap)  ## draw the cvImage
            ## self.display_img_on_label(self.cvImage) ## cannot use this function
            self.clip_rect.setWidth(self.ui.spinBox_ROI_Width.value())
            self.clip_rect.setHeight(self.ui.spinBox_ROI_Height.value())
            painter.drawRect(self.clip_rect)
            painter.setClipRect(self.clip_rect)
            self.update()
            self.ui.label_Image.update()
            #painter.end()
        
        if self.isSettingPosition:
            painter = PyQt5.QtGui.QPainter()
            painter.begin(self.ui.label_Image.pixmap())
            # pen = QtGui.QPen()
            # pen.setWidth(self.pen_params.width)
            # pen.setColor(QtGui.QColor(self.pen_params.color))
            #painter.setPen(pen)
            #painter.fillRect(event.rect(), QBrush(Qt.white))
            painter.setRenderHint(PyQt5.QtGui.QPainter.Antialiasing)
            painter.setPen(PyQt5.QtGui.QPen(PyQt5.QtGui.QBrush(PyQt5.Qt.blue), 1, PyQt5.Qt.DashLine))
            painter.drawPixmap(0, 0, self.pixMap)  ## draw the cvGoodImage
            
            self.draw_cross(painter, x= self.defect_x , y=self.defect_y, size = 10) ## draw a cross
            ## draw a cross
            self.update()
            self.ui.label_Image.update()

    def draw_cross(self, painter, x=100, y=100, size=5):
        painter.drawLine(x-size, y, x+size, y)
        painter.drawLine(x, y-size, x, y+size)
        return

    def getRect(self):
        return PyQt5.QtCore.QRect(self.clip_rect.topLeft(), PyQt5.QSize(self.roi_width, self.roi_height))  ## size 要改變


    def mousePressEvent(self, event):
        ## 修正滑鼠指標位置不一致的問題: 加入一個 point_shift
        point_shift = PyQt5.QtCore.QPoint()
        point_shift.setX(self.mouse_offset_x)  ## 加入位移
        point_shift.setY(self.mouse_offset_y)
        if self.isPainting:
            rect = self.getRect()
            #print(rect.x, rect.y, rect.width, rect.height)
            #print(event.pos)
            if rect.contains(event.pos()-point_shift):  ## 加入位移
                self.dragging = True
                self.drag_offset = rect.topLeft() - event.pos()
            else:
                self.dragging = None
        
        if self.isSettingPosition:
            ## Give the position of mouse press
            # self.defect_x = event.pos().x()
            # self.defect_y = event.pos().y()
            #point = self.ui.label_Image.mapFromGlobal(event.pos()) #nmapFromParent(event .pos())
            self.defect_x = event.pos().x() - self.mouse_offset_x #point.x() #self.ui.label_Image.mapFromParent  
            self.defect_y = event.pos().y()- self.mouse_offset_y  #point.y() #point.y() 
        
        return
    
    def mouseMoveEvent(self, event): 
        if self.isPainting:
            if self.dragging is None:
                return
            ## 修改最大的區域
            self.largest_rect = PyQt5.QtCore.QRect(0, 0, self.ui.label_Image.width(), self.ui.label_Image.height())
            left = self.largest_rect.left()
            right = self.largest_rect.right()
            top = self.largest_rect.top()
            bottom = self.largest_rect.bottom()
            
            point = event.pos() + self.drag_offset
            point.setX(max(left, min(point.x(), right)))
            point.setY(max(top, min(point.y(), bottom)))
            self.clip_rect.setTopLeft(point)
            point2 = event.pos() + self.drag_offset
            ## Tien: setX, Y: 決定 point2 的boundary，故舍 right -2，可以看得見
            point2.setX(min(right - 2, min(point2.x()+ self.roi_width, right))) #min(left+ right -50, min(point2.x()+100, right)))
            point2.setY(min(bottom - 2, min(point2.y()+self.roi_height, bottom)))
            self.ui.statusbar.showMessage("Point2:" + str(point2.x())+ ", " + str(point2.y()))
            self.clip_rect.setBottomRight(point2)
            
            if point.y() == top:
                point2.setY(top + self.roi_height)
                self.clip_rect.setBottomRight(point2)
            if point.x() == left:
                point2.setX(left + self.roi_width)
                self.clip_rect.setBottomRight(point2)
            if point2.x() == left + 400 -1:  ## double check this part
                point.setX(left + 300)
                self.clip_rect.setTopLeft(point)
            if point2.y() == (top + 400 -1): ## double check this part
                point.setY(left + 300)
                self.clip_rect.setTopLeft(point)
        #self.ui.label_Image.update()
        if self.isSettingPosition:
            point = self.ui.label_Image.mapFromGlobal(event.pos()) #nmapFromParent(event .pos())
            self.defect_x = event.pos().x() - self.mouse_offset_x #point.x() #self.ui.label_Image.mapFromParent  
            self.defect_y = event.pos().y() - self.mouse_offset_y # point.y() #point.y() 
            self.ui.statusbar.showMessage("Position: " + str(self.defect_x) + " , " + str(self.defect_y))
        self.update()
        return
    
    def mouseReleaseEvent(self, event):
        self.dragging = None
        return

    ######### HSV ##########################
    def set_hsv_by_image(self):
        #filename = QFileDialog.getOpenFileName(self.ccd_form, 'Open Image File', './')
        #if filename =="":
        #    return None
        #print("Open image file: ", filename)
        #cvImage = cv2.imread(filename[0])
        
        #hsv_seg = HSV_Class.HSV_Segment()
        self.dip.scale = 3
        self.dip.defect_name = "./SB_Defect_Bank/SB"
        self.dip.hsv_dynamic_segment(self.cvDefectImg) ## scaled defect image
        self.defect_image = self.dip.seg_image ## 
        self.display_img(self.cvDefectImg , self.ui.label_Defect_Image)
        self.hsv_low = self.dip.lower_hsv
        self.hsv_high =  self.dip.higher_hsv
        print("HSV bounds:", self.dip.lower_hsv, self.dip.higher_hsv)
        self.ui.spinBox_Hue_Low.setValue(self.hsv_low[0])
        self.ui.spinBox_Saturation_Low.setValue(self.hsv_low[1])
        self.ui.spinBox_Intensity_Low.setValue(self.hsv_low[2])
        self.ui.spinBox_Hue_High.setValue(self.hsv_high[0])
        self.ui.spinBox_Saturation_High.setValue(self.hsv_high[1])
        self.ui.spinBox_Intensity_High.setValue(self.hsv_high[2])      
        return self.defect_image

    def hsv_segment(self):
        if self.cvDefectImg is None:
            #print("Please load defect")
            QMessageBox.warning(self, "Warning", "Please load defect image first")
            return
        self.read_hsv_value()
        self.defect_image = self.dip.seg_image  = self.dip.hsv_segment(self.cvDefectImg, self.hsv_low, self.hsv_high)
        self.display_img(self.defect_image, self.ui.label_Defect_Image) #display_img(self.thresh , self.view)
        self.isSegmented = True
        return

    def read_hsv_value(self):
        self.hsv_low = [0, 0, 0]
        self.hsv_high = [179, 255, 255]
        self.hsv_low[0] = self.ui.spinBox_Hue_Low.value()
        self.hsv_low[1] = self.ui.spinBox_Saturation_Low.value()
        self.hsv_low[2] = self.ui.spinBox_Intensity_Low.value()
        self.hsv_high[0] = self.ui.spinBox_Hue_High.value()
        self.hsv_high[1] = self.ui.spinBox_Saturation_High.value()
        self.hsv_high[2] = self.ui.spinBox_Intensity_High.value()
        return
    def read_xyz_value(self):
        self.ui.lineEdit_X.setText(str(self.xyz_list[0]))
        self.ui.lineEdit_Y.setText(str(self.xyz_list[1]))
        self.ui.lineEdit_Z.setText(str(round(self.xyz_list[2],2)))

    def catch_all(self):
        
        for i in range(len(self.com_list)):
            if self.com_list[i] == str(int(self.det_list[i][5])):
                self.xyz_list[0] = (self.det_list[i][0] + self.det_list[i][2]) /2
                self.xyz_list[1] = (self.det_list[i][1] + self.det_list[i][3]) /2

            depth = self.rs.get_depth(int(self.xyz_list[0]), int(self.xyz_list[1]))
            self.xyz_list[2] = depth
            x = ((self.xyz_list[0]/ 1280) -0.575)
            y = (-0.46-(self.xyz_list[1]/1400))
            z = (self.xyz_list[2] / 7)
            URuse.UR_Control.move2obj(robot_ur5,x,y)
            time.sleep(2)
            URuse.UR_Control.grip_open(robot_ur5)
            time.sleep(2)
            URuse.UR_Control.catchobj(robot_ur5,z)
            time.sleep(2)
            URuse.UR_Control.grip_close(robot_ur5)
            time.sleep(2)
            URuse.UR_Control.move2obj(robot_ur5,x,y)
            time.sleep(2)
            URuse.UR_Control.move2final(robot_ur5)
            time.sleep(2)
            URuse.UR_Control.catchobj(robot_ur5,z)
            time.sleep(2)
            URuse.UR_Control.grip_open(robot_ur5)
            time.sleep(2)
            URuse.UR_Control.move2final(robot_ur5)
            time.sleep(2)
            URuse.UR_Control.move2pos(robot_ur5)
            time.sleep(5)
            URuse.UR_Control.grip_close(robot_ur5)
            time.sleep(1)

        
    def catch_obj(self):
        x = ((self.xyz_list[0]/ 1280) -0.575)
        y = (-0.46-(self.xyz_list[1]/1400))
        z = (self.xyz_list[2] / 7)
        URuse.UR_Control.move2obj(robot_ur5,x,y)
        time.sleep(2)
        URuse.UR_Control.grip_open(robot_ur5)
        time.sleep(2)
        print(self.angle)
        if self.angle >= -45:
            flag = 5
        else:
            flag = 2
        URuse.UR_Control.tcpmove(robot_ur5,flag,(abs(self.angle))/180*3.1415926)
        time.sleep(6)
        URuse.UR_Control.catchobj(robot_ur5,z)
        time.sleep(2)
        URuse.UR_Control.grip_close(robot_ur5)
        time.sleep(2)
        URuse.UR_Control.move2obj(robot_ur5,x,y)
        time.sleep(2)
        URuse.UR_Control.move2final(robot_ur5)
        time.sleep(2)
        URuse.UR_Control.catchobj(robot_ur5,z)
        time.sleep(2)
        URuse.UR_Control.grip_open(robot_ur5)
        time.sleep(2)
        URuse.UR_Control.move2final(robot_ur5)
        time.sleep(2)
        URuse.UR_Control.move2pos(robot_ur5)
        time.sleep(5)
        URuse.UR_Control.grip_close(robot_ur5)
        time.sleep(1)
    
    def load_params(self, filename = "./config/param_default.txt"):
        #filename = "./config/" + filename
        ## open a text file and write
        with open(filename, 'r') as f:
            lines = f.readlines()
            #print(lines)
            line = lines[0].strip().split(",")
            #print(line)
            ## load hsv
            #self.hsv_low = [0, 0, 0]
            #self.hsv_high = [179, 255, 255]
            #self.ui.spinBox_Hue_Low.setValue(int(line[0]))
            # self.hsv_low[0] = self.ui.spinBox_Hue_Low.value()
            # self.ui.spinBox_Saturation_Low.setValue(int(line[1]))
            # self.hsv_low[1] = self.ui.spinBox_Saturation_Low.value()
            # self.ui.spinBox_Intensity_Low.setValue(int(line[2]))
            # self.hsv_low[2] = self.ui.spinBox_Intensity_Low.value()
            
            # line = lines[1].strip().split(",")
            #print(line)
            # self.ui.spinBox_Hue_High.setValue(int(line[0]))
            # self.hsv_high[0] = self.ui.spinBox_Hue_High.value()            
            # self.ui.spinBox_Saturation_High.setValue(int(line[1]))
            # self.hsv_high[1] = self.ui.spinBox_Saturation_High.value()
            # self.ui.spinBox_Intensity_High.setValue(int(line[2]))
            # self.hsv_high[2] = self.ui.spinBox_Intensity_High.value()

            # line = lines[2].strip().split(",")
#             #print(line)
            # self.ui.spinBox_Scale.setValue(int(line[0]))
            # self.ui.spinBox_ROI_Width.setValue(int(line[1]))
            # self.ui.spinBox_ROI_Height.setValue(int(line[2]))
            
            # line = lines[3].strip()
            # self.ui.spinBox_Scale_Transfer.setValue(int(line))
            # line = lines[4].strip()
            # self.ui.spinBox_Scale_Defect_Image.setValue(int(line))
            # self.ui.statusbar.showMessage("Load params: " + filename)
        return

    def save_params(self, filename = "./config/param_default.txt"):
        ## get product name
        #product_name = self.ui.lineEdit_HSV.text()
        ##print(product_name)
        ## if not exist, create a directory
        # path = "./config/products/" + product_name
        # if os.path.isdir(path):
        #     print("Product %s exist", product_name)
        # else:
        #     print("Create a product ")
        #     try:
        #         os.mkdir(path)
        #     except OSError:
        #         print ("Creation of the directory %s failed" % path)
        #         return
        #     else:
        #         print ("Successfully created the directory %s " % path)

        ## open a text file and write
        with open(filename, 'w') as f:
            f.write(str(self.hsv_low[0]) + "," + str(self.hsv_low[1])+ "," + str(self.hsv_low[2])+ "\n")
            f.write(str(self.hsv_high[0]) + "," + str(self.hsv_high[1])+ "," + str(self.hsv_high[2])+ "\n") 

            scale = str(self.ui.spinBox_Scale.value())
            roi_width = str(self.ui.spinBox_ROI_Width.value())
            roi_height = str(self.ui.spinBox_ROI_Height.value())
            f.write(scale + "," + roi_width + ", " + roi_height + "\n")  ## kernel size

            scale_good = str(self.ui.spinBox_Scale_Transfer.value())
            scale_defect = str(self.ui.spinBox_Scale_Defect_Image.value())
            f.write(scale_good + "\n")
            f.write(scale_defect + "\n")
            # min_area = self.ccd_form_ui.spinBox_Min_Area.value()
            # f.write(str(min_area) + "\n")  ## Min blob size
            
            self.ui.statusbar.showMessage("Save params: " + filename)
        return
    def gettcp(self): #取得TCP設定值
        t1 = 0.001*int(self.ui.plainTextEdit_x.toPlainText())
        t2 = 0.001*int(self.ui.plainTextEdit_y.toPlainText())
        t3 = 0.001*int(self.ui.plainTextEdit_z.toPlainText())
        t4 = int(self.ui.plainTextEdit_rx.toPlainText())
        t5 = int(self.ui.plainTextEdit_ry.toPlainText())
        t6 = int(self.ui.plainTextEdit_rz.toPlainText())
        var = t1,t2,t3,t4,t5,t6
        URuse.UR_Control.settcp(robot_ur5,var)
        self.ui.label_37.setText('設定成功')
        return 

    def grip(self): #爪子
        self.ui.pushButton_gripopen.clicked.connect(lambda : URuse.UR_Control.grip_open(robot_ur5))
        self.ui.pushButton_gripclose.clicked.connect(lambda : URuse.UR_Control.grip_close(robot_ur5))
        return

    def movelvalue(self): #得到XY數據, pos 
        pos = URuse.UR_Control.Getdata(robot_ur5)
        postcp = URuse.UR_Control.Gettcppose(robot_ur5)
        text = self.ui.comboBox_list.currentText()
        
        if text == '視角':
            self.ui.label_x_2.setText(str(round(1000*pos[0]-6,2)))
            self.ui.label_y_2.setText(str(round(pos[1]*1000,2)))
            self.ui.label_z_2.setText(str(round(pos[2]*1000-400,2)))
            self.ui.label_rx_2.setText(str(round(pos[3],3)))
            self.ui.label_ry_2.setText(str(round(pos[4]+0.015,3)))
            self.ui.label_rz_2.setText(str(round(pos[5]-0.015,3)))
        elif text == '機座':
            self.ui.label_x_2.setText(str(round(1000*pos[0],2)))
            self.ui.label_y_2.setText(str(round(pos[1]*1000,2)))
            self.ui.label_z_2.setText(str(round(pos[2]*1000,2)))
            self.ui.label_rx_2.setText(str(round(pos[3],3)))
            self.ui.label_ry_2.setText(str(round(pos[4],3)))
            self.ui.label_rz_2.setText(str(round(pos[5],3)))
        elif text == '工具':
            self.ui.label_x_2.setText(str(round(postcp[0])))
            self.ui.label_y_2.setText(str(round(postcp[1])))
            self.ui.label_z_2.setText(str(round(postcp[2])))
            self.ui.label_rx_2.setText(str(round(postcp[3])))
            self.ui.label_ry_2.setText(str(round(postcp[4])))
            self.ui.label_rz_2.setText(str(round(postcp[5])))
        return 

    def movejvalue(self): #取得joint 1rad = 57.29578 degree
        posej = URuse.UR_Control.Getdataj(robot_ur5)
        self.ui.label_4.setText(str(round(posej[0]*180/3.14,2)))
        self.ui.label_6.setText(str(round(posej[1]*180/3.14,2)))
        self.ui.label_7.setText(str(round(posej[2]*180/3.14,2)))
        self.ui.label_8.setText(str(round(posej[3]*180/3.14,2)))
        self.ui.label_9.setText(str(round(posej[4]*180/3.14,2)))
        self.ui.label_10.setText(str(round(posej[5]*180/3.14,2)))        
        return

    def setvaluepage(self): #連接到設定數值葉面
        self.ui.pushButton_2.clicked.connect(lambda : self.mpos())
        return

    def updataData(self): #更新數據
        self.timer=QTimer(self) # 呼叫 QTimer 
        self.timer.timeout.connect(self.movelvalue) #當時間到時會執行
        self.timer.timeout.connect(self.movejvalue)
        self.timer.start(1000) #啟動 Timer 每隔1000ms 1秒 會觸發 
        return

    def movel(self): #線性移動
        text = self.ui.comboBox_list.currentText()
        if text == '工具':
            print(text)
            self.ui.pushButton_tcpdown.pressed.connect(lambda : self.toolx())
            self.ui.pushButton_tcpup.pressed.connect(lambda : self.toolxx())
            self.ui.pushButton_tcpleft.pressed.connect(lambda : self.toolyy())
            self.ui.pushButton_tcpright.pressed.connect(lambda : self.tooly())
            self.ui.pushButton_up_2.pressed.connect(lambda : self.toolz())
            self.ui.pushButton_down.pressed.connect(lambda : self.toolzz())
        else:
            self.ui.pushButton_tcpup.pressed.connect(lambda : self.movelinexx())
            self.ui.pushButton_tcpdown.pressed.connect(lambda : self.movelinex())
            self.ui.pushButton_tcpleft.pressed.connect(lambda : self.movelineyy())
            self.ui.pushButton_tcpright.pressed.connect(lambda : self.moveliney())
            self.ui.pushButton_up_2.pressed.connect(lambda : self.movelinez())
            self.ui.pushButton_down.pressed.connect(lambda : self.movelinezz())
        #停止
        self.ui.pushButton_tcpup.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_tcpdown.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_tcpleft.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_tcpright.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_up_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_down.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        return
    def mpos(self):


        URuse.UR_Control.move2pos(robot_ur5)
        
##        while 1:
##            posej = URuse.UR_Control.Getdataj(robot_ur5)
##            print(round(posej[0]*180/3.14,2))
##            if a < (round(posej[0]*180/3.14,2)):
##                URuse.UR_Control.MoveJoint(robot_ur5, 6,1)
##            else:        
##                URuse.UR_Control.stoprob(robot_ur5)
##                break
##        while 1:
##            posej = URuse.UR_Control.Getdataj(robot_ur5)
##            print(round(posej[2]*180/3.14,2))
##            if c < (round(posej[2]*180/3.14,2)):
##                URuse.UR_Control.MoveJoint(robot_ur5, 8,1)
##            else:        
##                URuse.UR_Control.stoprob(robot_ur5)
##                break
##        while 1:
##            posej = URuse.UR_Control.Getdataj(robot_ur5)
##            print(round(posej[1]*180/3.14,2))
##            if b < (round(posej[1]*180/3.14,2)):
##                URuse.UR_Control.MoveJoint(robot_ur5, 7,1)
##            else:        
##                URuse.UR_Control.stoprob(robot_ur5)
##                break
##        
##        while 1:
##            posej = URuse.UR_Control.Getdataj(robot_ur5)
##            print(round(posej[3]*180/3.14,2))
##            if d < (round(posej[3]*180/3.14,2)):
##                URuse.UR_Control.MoveJoint(robot_ur5, 9,1)
##            else:        
##                URuse.UR_Control.stoprob(robot_ur5)
##                break
##        while 1:
##            posej = URuse.UR_Control.Getdataj(robot_ur5)
##            print(round(posej[4]*180/3.14,2))
##            if e > (round(posej[4]*180/3.14,2)):
##                URuse.UR_Control.MoveJoint(robot_ur5, 4,1)
##            else:        
##                URuse.UR_Control.stoprob(robot_ur5)
##                break


    def movelinex(self):
        var = 0
        URuse.UR_Control.moveline(robot_ur5,var)
        return

    def moveliney(self):
        var = 1
        URuse.UR_Control.moveline(robot_ur5,var)
        return

    def movelinez(self):
        var = 2
        URuse.UR_Control.moveline(robot_ur5,var)
        return

    def movelinexx(self):
        var = 3
        URuse.UR_Control.moveline(robot_ur5,var)
        return

    def movelineyy(self):
        var = 4
        URuse.UR_Control.moveline(robot_ur5,var)
        return
    
    def movelinezz(self):
        var = 5
        URuse.UR_Control.moveline(robot_ur5,var)
        return

    def toolx(self):
        va = 0
        URuse.UR_Control.toolmovel(robot_ur5,va)
        return

    def tooly(self):
        va = 1
        URuse.UR_Control.toolmovel(robot_ur5,va)
        return

    def toolz(self):
        va = 2
        URuse.UR_Control.toolmovel(robot_ur5,va)
        return

    def toolxx(self):
        va = 3
        URuse.UR_Control.toolmovel(robot_ur5,va)
        return

    def toolyy(self):
        va = 4
        URuse.UR_Control.toolmovel(robot_ur5,va)
        return

    def toolzz(self):
        va = 5
        URuse.UR_Control.toolmovel(robot_ur5,va)
        return

    def MoveJoint(self): #六軸移動, number
        self.ui.pushButton_base.pressed.connect(lambda : self.movebase2())
        self.ui.pushButton_base2.pressed.connect(lambda : self.movebase())
        self.ui.pushButton_shouder.pressed.connect(lambda : self.moveshoulder2())
        self.ui.pushButton_shoulder2.pressed.connect(lambda : self.moveshoulder())
        self.ui.pushButton_elbow.pressed.connect(lambda : self.moveelbow2())
        self.ui.pushButton_elbow2.pressed.connect(lambda : self.moveelbow())
        self.ui.pushButton_wrist1.pressed.connect(lambda : self.movewrist12())
        self.ui.pushButton_wrist12.pressed.connect(lambda : self.movewrist1())
        self.ui.pushButton_wrist2.pressed.connect(lambda : self.movewrist22())
        self.ui.pushButton_wrist22.pressed.connect(lambda : self.movewrist2())
        self.ui.pushButton_wrist3.pressed.connect(lambda : self.movewrist32())
        self.ui.pushButton_wrist32.pressed.connect(lambda : self.movewrist3())
        #拜託停下來
        self.ui.pushButton_base.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_base2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_shouder.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_shoulder2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_elbow.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_elbow2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist1.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist12.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist22.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist3.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist32.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        return


    def movebase(self):
        var = 0
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def moveshoulder(self):
        var = 1
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return

    def moveelbow(self):
        var = 2
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return

    def movewrist1(self):
        var = 3
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return
    
    def movewrist2(self):
        var = 4
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return

    def movewrist3(self):
        var = 5
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return

    def movebase2(self):
        var = 6
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return

    def moveshoulder2(self):
        var = 7
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return

    def moveelbow2(self):
        var = 8
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return

    def movewrist12(self):
        var = 9
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return
    
    def movewrist22(self):
        var = 10
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return

    def movewrist32(self):
        var = 11
        URuse.UR_Control.MoveJoint(robot_ur5,var)
        return

    def movetcp(self): #TCP方向
        self.ui.pushButton_movedown.pressed.connect(lambda :self.movery())
        self.ui.pushButton_moveup.pressed.connect(lambda :self.moveryback())
        self.ui.pushButton_moveright.pressed.connect(lambda :self.moverxback())
        self.ui.pushButton_moveleft.pressed.connect(lambda :self.moverx())
        self.ui.pushButton_movep.pressed.connect(lambda :self.moverz())
        self.ui.pushButton_movep_2.pressed.connect(lambda :self.moverzback())
        #拜託停下來
        self.ui.pushButton_moveup.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_movedown.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_moveright.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_moveleft.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_movep.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_movep_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))   
        return
    
    def moverx(self):
        var = 0
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def movery(self):
        var = 1
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return
    
    def moverz(self):
        var = 2
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def moverxback(self):
        var =3
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def moveryback(self):
        var = 4
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def moverzback(self):
        var = 5
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def settohome(self): #回零
        self.ui.pushButton_home.clicked.connect(lambda: URuse.UR_Control.set2Home(robot_ur5))
        return

    def URspeed(self): #調整速度
        self.ui.horizontalSlider_speed.setRange(1,100)
        #speed = self.ui.horizontalSlider_speed.value()
        self.ui.label_speedvalue.setText("30")
        speed = self.ui.horizontalSlider_speed.value
        self.ui.horizontalSlider_speed.valueChanged.connect(lambda :self.speedvalue())
        self.ui.horizontalSlider_speed.valueChanged.connect(lambda :URuse.UR_Control.speed(robot_ur5,speed))
        return speed
        
    def speedvalue(self): 
        speed = self.ui.horizontalSlider_speed.value
        self.ui.label_speedvalue.setText(str(speed))
        return 
 
    def freedrive(self): #自由移動
        self.ui.pushButton_free.pressed.connect(lambda : URuse.UR_Control.free(robot_ur5))
        self.ui.pushButton_free.released.connect(lambda: URuse.UR_Control.stoprob(robot_ur5))
        return

    def stop(self): #全部暫停
        self.ui.pushButton.clicked.connect(lambda: URuse.UR_Control.stoprob(robot_ur5))
        return

    def closeEvent(self, event):
        """
        重写closeEvent方法，实现dialog窗体关闭时执行一些代码
        :param event: close()触发的事件
        :return: None
        """
        reply = PyQt5.QtWidgets.QMessageBox.question(self,
                                               'iVi Lab',
                                               "是否要退出系統？",
                                               PyQt5.QtWidgets.QMessageBox.Yes | PyQt5.QtWidgets.QMessageBox.No,
                                               PyQt5.QtWidgets.QMessageBox.No)
        if reply == PyQt5.QtWidgets.QMessageBox.Yes:
            if self.rs is not None:
                self.rs.quit_rs()
            event.accept()
        else:
            event.ignore()
    
                
class Subwindow(QWidget): #數據輸入子視窗
    def __init__(self, parent = iViLab_Main):
        super(Subwindow, self).__init__()
        self.sub = Ui_Form()
        self.sub.setupUi(self)
        self.entery()
        self.setWindowTitle('Enter Value')

        #combo
        combo=['機座']
        self.sub.comboBox_set.addItems(combo)
        self.sub.pushButton_setcancel.clicked.connect(lambda :self.setcancel())
        self.sub.pushButton_setOK.clicked.connect(lambda : self.gettext())
    
    def subopen(self):
        self.show()
        return

    def setcancel(self): #視窗關閉
        self.close()
        return
        
    def entery(self): #數值顯示
        pos = URuse.UR_Control.Getdata(robot_ur5)
        posej = URuse.UR_Control.Getdataj(robot_ur5)
        postcp = URuse.UR_Control.Gettcppose(robot_ur5)
        self.sub.plainTextEdit_x.setPlainText(str(round(1000*pos[0],2)))
        self.sub.plainTextEdit_y.setPlainText(str(round(1000*pos[1],2)))
        self.sub.plainTextEdit_z.setPlainText(str(round(1000*pos[2],2)))
        self.sub.plainTextEdit_rx.setPlainText(str(round(-pos[3],3)))
        self.sub.plainTextEdit_ry.setPlainText(str(round(-pos[4],3)))
        self.sub.plainTextEdit_rz.setPlainText(str(round(pos[5],3)))

        self.sub.plainTextEdit_base.setPlainText(str(round(posej[0]*180/3.14,2)))
        self.sub.plainTextEdit_shoulder.setPlainText(str(round(posej[1]*180/3.14,2)))
        self.sub.plainTextEdit_elbow.setPlainText(str(round(posej[2]*180/3.14,2)))
        self.sub.plainTextEdit_wrist1.setPlainText(str(round(posej[3]*180/3.14,2)))
        self.sub.plainTextEdit_wrist2.setPlainText(str(round(posej[4]*180/3.14,2)))
        self.sub.plainTextEdit_wrist3.setPlainText(str(round(posej[5]*180/3.14,2)))
        return

    def gettext(self): #得到數值
        x = self.sub.plainTextEdit_x.toPlainText()
        y = self.sub.plainTextEdit_y.toPlainText()
        z = self.sub.plainTextEdit_z.toPlainText()
        rx = self.sub.plainTextEdit_rx.toPlainText()
        ry = self.sub.plainTextEdit_ry.toPlainText()
        rz = self.sub.plainTextEdit_rz.toPlainText()
        b = self.sub.plainTextEdit_base.toPlainText()
        s =self.sub.plainTextEdit_shoulder.toPlainText()
        e = self.sub.plainTextEdit_elbow.toPlainText()
        w1 = self.sub.plainTextEdit_wrist1.toPlainText()
        w2 = self.sub.plainTextEdit_wrist2.toPlainText()
        w3 = self.sub.plainTextEdit_wrist3.toPlainText()
        
        print('gettext',x,y,z,rx,ry,rz,b,s,e,w1,w2,w3)
        return x,y,z,rx,ry,rz,b,s,e,w1,w2,w3
        

##def main():
##    """
##    主函数，用于运行程序
##    :return: None
##    """
##    app = PyQt5.QtWidgets.QApplication(sys.argv)
##    main =   # 注意修改为了自己重写的Dialog类
##    #ui = Ui_Main()
##    #ui.setupUi(main)
##    #main.show()  # 显示了自己重写的Dialog类
##    sys.coinit_flags = 2
##    sys.exit(app.exec_())
##
##
##if __name__ == '__main__':
##    main()
    



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = iViLab_Main()
    main.show()
    sys.exit(app.exec_())
    count = 0
    while (count < 1):
        time.sleep(2)

        count = count + 1
        print("The count is:", count)
    print("Program finish")
data = r.recv(1024)
r.close()

