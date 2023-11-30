import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget,QDialog, QLabel, QTextEdit, QTextBrowser, QHBoxLayout, QVBoxLayout,QPushButton
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from PyQt5.QtCore import QTimer
import time
import threading
import urx
import math
import socket
from View.UR_ui_5 import * #連接畫面
from View.setvalue import * #設數值小視窗
from View.program1 import * #program輸入數值
import URuse
import URvis


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
cap_ur5 = URvis.UR_cap()

class URpractice(QMainWindow):
    def __init__(self, parent = None):
        super(URpractice, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('UR Practice')
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
        self.ccd() #開鏡頭
        self.dia1 = SubDialog()
        self.sub = Subwindow()
        self.show()

        #combolist set
        combo = ['視角','機座','工具']
        self.ui.comboBox_list.addItems(combo)
        self.ui.comboBox_list.currentTextChanged.connect(self.movelvalue)
        self.ui.comboBox_list.currentTextChanged.connect(self.movel)
        self.ui.pushButton_3.clicked.connect(lambda : self.gettcp())
    
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
        self.ui.pushButton_2.clicked.connect(lambda : self.sub.subopen())
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
    
    def ccd(self): #開啟鏡頭
        self.ui.pushButton_11.clicked.connect(lambda : URvis.UR_cap.cap(cap_ur5))
        self.ui.pushButton_13.clicked.connect(lambda :self.cap1())
        self.ui.pushButton_14.clicked.connect(lambda : self.cap2())
        #self.ui.pushButton_16.clicked.connect(lambda : self.dia1.calldia())
        self.ui.pushButton_15.clicked.connect(lambda :self.cap3())
        #方向
        self.ui.pushButton_moveup_2.pressed.connect(lambda :self.movery())
        self.ui.pushButton_movedown_2.pressed.connect(lambda :self.moveryback())
        self.ui.pushButton_moveright_2.pressed.connect(lambda :self.moverx())
        self.ui.pushButton_moveleft_2.pressed.connect(lambda :self.moverxback())
        self.ui.pushButton_movep_4.pressed.connect(lambda :self.moverz())
        self.ui.pushButton_movep_3.pressed.connect(lambda :self.moverzback())
        #線性移動
        self.ui.pushButton_tcpup_2.pressed.connect(lambda : self.movelineyy())
        self.ui.pushButton_tcpdown_2.pressed.connect(lambda : self.moveliney())
        self.ui.pushButton_tcpleft_2.pressed.connect(lambda : self.movelinexx())
        self.ui.pushButton_tcpright_2.pressed.connect(lambda : self.movelinex())
        self.ui.pushButton_up_3.pressed.connect(lambda : self.movelinezz())
        self.ui.pushButton_down_2.pressed.connect(lambda : self.movelinez())
        #停止
        self.ui.pushButton_tcpup_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_tcpdown_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_tcpleft_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_tcpright_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_up_3.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_down_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_moveup_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_movedown_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_moveright_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_moveleft_2.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_movep_4.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_movep_3.released.connect(lambda : URuse.UR_Control.stoprob(robot_ur5)) 
        return

    def cap1(self):
        call = URvis.UR_cap.x_y_center(cap_ur5)
        if call == True:
            self.dia1.calldia()
            num = self.dia1.get()
        else:
            num = None
            print("not Detected")
        return num

    def cap2(self):
        num = self.dia1.get()
        pos = URuse.UR_Control.Getdata(robot_ur5)
        x,y = URvis.UR_cap.pixel_distance(cap_ur5,pos[0],pos[1],num)
        x =list(dict(x=x,y=y).keys())[0]
        y = list(dict(x=x,y=y).keys())[1]
        self.ui.label_29.setText(str(round(x,2)))
        self.ui.label_30.setText(str(round(y,2)))
        z = URuse.UR_Control.robdis(robot_ur5)
        self.ui.label_31.setText(str(round(z*100,2)))
        return x,y

    def cap3(self):
        num = self.dia1.get()
        x,y = self.cap2()
        pos = URuse.UR_Control.Getdata(robot_ur5)
        URuse.UR_Control.moveto(robot_ur5,x,y,num)
        return


class SubDialog(QDialog): #物件選擇
    def __init__(self, parent = URpractice):
        super(SubDialog, self).__init__()
        self.dia1 = Ui_Dialog()
        self.dia1.setupUi(self)
        self.setWindowTitle("選擇抓取物件")
        self.dia1.pushButton_diaok.clicked.connect(lambda : self.get())
        self.dia1.pushButton_3.clicked.connect(lambda : self.buttoncancel())

    def calldia(self): #呼叫其他自定義訊息框
        image = QtGui.QPixmap('C:/Users/user/Desktop/choose.jpg')
        self.dia1.label_3.setPixmap(image)
        self.show()
        return

    def buttoncancel(self): #關閉訊息框
        num = None
        self.close()
        return num

    def get(self): #取得數值
        num = int(self.dia1.plainTextEdit.toPlainText())
        self.close()
        #x,y = URvis.UR_cap.pixel_distance(cap_ur5,pos[0],pos[1],num)
        return num

class Subwindow(QWidget): #數據輸入子視窗
    def __init__(self, parent = URpractice):
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
        self.sub.close()
        print('gettext',x,y,z,rx,ry,rz,b,s,e,w1,w2,w3)
        return x,y,z,rx,ry,rz,b,s,e,w1,w2,w3

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = URpractice()
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
