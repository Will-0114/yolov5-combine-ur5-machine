import cv2
import numpy as np
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import matplotlib.pyplot as plt
import math
import threading
import time
import math3d as m3d
import URvis
capur5 = URvis.UR_cap()

class UR_Control():
    def __init__(self):
        self.v = 0.3
        self.a = 0.5
        self.Pi = math.pi
        self.jointspeed = 1
        self.pos_bool = False
        self.rob = urx.Robot("192.168.0.3")
        self.Gripper = Robotiq_Two_Finger_Gripper(self.rob)
        self.r = 0.05
        self.l = 0
        
    def speed(self,vel):
        self.v = vel*0.01
        return 

    def settcp(self,var):
        self.rob.set_tcp(var)
        return

    def Connect_ur(self):
        self.Gripper = Robotiq_Two_Finger_Gripper(self.rob)
        return

    def Disconnect_ur(self):
        self.rob.stopj(self.a)
        self.rob.stopl(1)
        self.rob.close()
        print("Robot is close")
        return

    def Gettcppose(self):
        post = self.rob.getl()
        return post

    def Getdata(self):
        pos = self.rob.getl()
        return pos

    def Getdataj(self):
        posej = self.rob.getj()
        return posej

    def set2Home(self):
        self.rob.movej([0, -(self.Pi/2), 0, -(self.Pi/2), 0, 0], acc=0.3, vel=0.05, wait=False)
        time.sleep(1)
        return

        
    def grip_open(self):
        self.Gripper.open_gripper()
        return

    def grip_close(self):
        self.Gripper.close_gripper()
        return

    def MoveJoint(self, number): #六軸
        if self.pos_bool == False:
            pose = self.rob.getj()
            self.posej = pose
            print("Get Position")
        else:
            print("Already Get")
        if number <6:
            print(number)
            self.pos_bool = True
            a = 1
            self.posej[number] += a*self.jointspeed
            self.rob.movej(self.posej, vel=self.v, wait = False)
            
        #    self.jointspeed +=1
        else:
            number -= 6
            self.pos_bool = True
            a = 1
            self.posej[number] -= a*self.jointspeed
            self.rob.movej(self.posej, vel=self.v, wait = False)
            
        #    self.jointspeed +=1
        pose = self.rob.getj()
        self.posej = pose
        print(pose)
        time.sleep(1)
        self.jointspeed = 1
        print('wait for 1 seconds')
        return
    def MoveJoint(self, number,a): #六軸
        if self.pos_bool == False:
            pose = self.rob.getj()
            self.posej = pose
            print("Get Position")
        else:
            print("Already Get")
        if number <6:
            print(number)
            self.pos_bool = True
            self.posej[number] += 5*self.jointspeed
            self.rob.movej(self.posej, vel=self.v, wait = False)
            
        #    self.jointspeed +=1
        else:
            number -= 6
            self.pos_bool = True
            self.posej[number] -= 5*self.jointspeed
            self.rob.movej(self.posej, vel=self.v, wait = False)
            
        #    self.jointspeed +=1
        a= 0
        pose = self.rob.getj()
        self.posej = pose
        print(pose)
        time.sleep(1)
        self.jointspeed = 1
        print('wait for 1 seconds')
        return

    def stoprob(self): #停止機器人
        self.rob.stopj(self.a)
        self.rob.stopl(1)
        return
    def move2pos(self):
        self.rob.movej([0.6202548146247864, -1.1531885305987757, 1.1349406242370605, -1.5446456114398401, -1.6085646788226526, -0.9475105444537562], acc=0.3, vel=0.3, wait=False)
        time.sleep(4)
        #self.rob.movel([0.41250992, -0.52857208, 0.4014003184567244, -0.0026672071252298794, -3.1226434885300725, -0.05995872436838651], acc=0.3, vel=0.3, wait=False)
        #posel = self.rob.getl()
        #print(posel)
        
        time.sleep(1)
        
        return
    def move2angle(self,angle):
        posej = self.rob.getj()
        posej[5] = angle
        self.rob.movej(posej, acc=0.3, vel=0.3, wait=False)
        time.sleep(4)
    
    def move2final(self):
        self.rob.movel([0.2758084026719253, -0.46057151886938635, 0.4014003184567244, -0.0026672071252298794, -3.1226434885300725, -0.05995872436838651], acc=0.3, vel=0.3, wait=False)
        #time.sleep(4)
        
        
        time.sleep(1)
        
        return
    
    def move2obj(self,x,y):
        self.rob.movel([x, y, 0.40137838151974325, -0.0028287375388886924, -3.1227232310462356, -0.06018246184371976], acc =self.a, vel = self.v, wait = False)
        time.sleep(4)
        posel = self.rob.getl()
        print(posel)
        return

    def catchobj(self,z):
        posel = self.rob.getl()
        posel[2] = z
        self.rob.movel(posel, acc =self.a, vel = self.v, wait = False)
        time.sleep(4)
        return
    
    def moveline(self, number): #線性移動
        self.posel = self.rob.getl()
        if number<3:
            self.posel[number] += 0.005
            self.rob.movel(self.posel, acc =self.a, vel = self.v, wait = False)
        else:
            number -=3
            self.posel[number] -= 0.005
            self.rob.movel(self.posel, acc =self.a, vel = self.v, wait = False)
        time.sleep(1)
        print(self.posel)
        return

    def movep(self, number): #圓周移動
        posep = self.rob.getl()
        self.posep = posep
        if number <3: 
            pose = self.rob.getl(wait = True)
            self.posep[number] += self.jointspeed
            self.rob.movep(self.posep, acc=self.a, vel=self.v, radius = self.r, wait = False)
            while True:
                p = self.rob.getl(wait = True)
                if p[number] > self.posep[number]-0.05:
                    break
            #self.rob.movep(self.posep, acc = self.a, vel = self.v, radius = 0 , wait = True)

        else:
            number -= 2
            self.posep[number] -= self.jointspeed
            self.rob.movep(self.posep, acc = self.a, vel=self.v, radius = self.r, wait = False)
            while True:
                p = self.rob.getl(wait = True)
                if p[number] < self.posep[number]+0.05:
                    break
            #self.rob.movep(self.posep, acc = self.a, vel = self.v, radius = 0 , wait = True)
        time.sleep(2)
        print('wait for 2 seconds', self.posep)
        return

    def toolmovel(self, number): #工具模式的movel
        if number ==0:
            self.l +=0.005
            self.rob.translate_tool((self.l, 0, 0), acc=self.a, vel=self.v, wait = False)
        elif number == 1:
            self.l +=0.005
            self.rob.translate_tool((0, self.l, 0), acc=self.a, vel=self.v, wait = False)
        elif number ==2:
            self.l +=0.005
            self.rob.translate_tool((0, 0, self.l), acc=self.a, vel=self.v, wait = False)
        elif number == 3:
            self.l -=0.005
            self.rob.translate_tool((self.l, 0, 0), acc=self.a, vel=self.v, wait = False)
        elif number == 4:
            self.l -=0.005
            self.rob.translate_tool((0, self.l, 0), acc=self.a, vel=self.v, wait = False)
        elif number == 5:
            self.l -=0.005
            self.rob.translate_tool((0, 0, self.l), acc=self.a, vel=self.v, wait = False)
        time.sleep(1)
        print('movetcpl success')
        return
              
    def free(self): #自由操作
        self.rob.set_freedrive(1,30)
        return

    def tcpmove(self, number): #TCP方向
        a = (self.Pi/180)*5
        trans = self.rob.get_pose()
        if number == 0:
            trans.orient.rotate_xt(a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05, wait = False) 
        elif number ==1:
            trans.orient.rotate_yt(a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05 ,wait = False)
        elif number ==2:
            trans.orient.rotate_zt(a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05 ,wait = False)
        elif number == 3 :
            trans.orient.rotate_xt(-a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05, wait = False) 
        elif number == 4 :
            trans.orient.rotate_yt(-a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05, wait = False)
        elif number == 5 :
            trans.orient.rotate_zt(-a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05, wait = False)  
        time.sleep(1)
        return
    
    def tcpmove(self, number,a): #TCP方向
        trans = self.rob.get_pose()
        if number == 0:
            trans.orient.rotate_xt(a)
            self.rob.set_pose(trans, acc=0.2, vel=0.2, wait = False) 
        elif number ==1:
            trans.orient.rotate_yt(a)
            self.rob.set_pose(trans, acc=0.2, vel=0.2 ,wait = False)
        elif number ==2:
            trans.orient.rotate_zt(a)
            self.rob.set_pose(trans, acc=0.2, vel=0.2 ,wait = False)
        elif number == 3 :
            trans.orient.rotate_xt(-a)
            self.rob.set_pose(trans, acc=0.2, vel=0.2, wait = False) 
        elif number == 4 :
            trans.orient.rotate_yt(-a)
            self.rob.set_pose(trans, acc=0.2, vel=0.2, wait = False)
        elif number == 5 :
            trans.orient.rotate_zt(-a)
            self.rob.set_pose(trans, acc=0.2, vel=0.2, wait = False)  
        time.sleep(1)
        return

    def testmovec(self): #movec測試
        print("Test movec")
        pose = self.rob.get_pose()
        via = pose.copy()
        via.pos[0] += l
        to = via.copy()
        to.pos[1] += l
        self.rob.movec(via, to, acc=a, vel=v)
        return

    def robdis(self):
        a, b =self.rob.get_analog_inputs()
        self.rob.get_analog_inputs()
        dis = 21.5 + a*4.024 #a無法計算 須改
        print('a,dis:',a,dis)
        zz= (dis*0.01)-0.1
        print('zz:',zz)
        return zz

    def moveto(self,x,y,num):
        zz = self.robdis()
        pos = self.rob.getl()
        #x,y = URvis.UR_cap.pixel_distance(capur5,pos[0],pos[1],num)
        print (x,y,zz)
        print(pos)
        pos[0]+=x*0.01
        pos[1]+=y*0.01
        pos[2]-=zz
        print(pos)
        
        #self.grip_open()
        #time.sleep(1)
        #self.rob.movel((x,y,pos[2],pos[3],pos[4],pos[5]), acc =0.2, vel = 0.2, wait = False) 
        #time.sleep(2)
        #pos[2]-=zz
        #self.rob.movel(pos,acc=0.2,vel = 0.3, wait = false)
        #time.sleep(2)

        #URvis.UR_cap.cap00(capur5)
        #self.grip_close()
        time.sleep(1)

        #[2]+=zz
        #self.rob.movel(pos,acc=0.2,vel = 0.3, wait = false)
        return 


    


