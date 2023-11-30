## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseL515(object):
    pipeline = rs.pipeline()
    config = rs.config()
    Stop = True
    isActive = False
    def __init__(self):
        self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color,1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.infrared,1024, 768, rs.format.y8, 30)
        #self.pipeline.start(self.config)
        return

    def start_rs(self):
        self.pipeline.start(self.config)
        self.isActive = True
        return

    def display_cv(self):
        self.Stop = False
        # try:  ## after stable, please put this back
        while not self.Stop:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame()
            if not depth_frame or not color_frame:
                continue
            print(self.get_depth(100, 100))
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            #images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense: depth', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('RealSense: color', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('RealSense: infrared', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense: depth', depth_colormap)
            cv2.imshow('RealSense: color', color_image)
            cv2.imshow('RealSense: infrared', ir_image)
            key = cv2.waitKey(1)
            if key == 27:
                break
        # except:
        #     print("Abnormal ..., please reconnect the system ")
        return

    def get_depth_image(self, isShow = False):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        if isShow:
            cv2.imshow('RealSense: depth', depth_colormap)
            cv2.waitKey(0)
        #print("Under construction...")
        return depth_colormap
    
    def get_color_image(self, isShow = False):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if isShow:
            cv2.imshow('RealSense: depth', color_image)
            cv2.waitKey(0)
        #print("Under construction...")
        return color_image
    
    def get_ir_image(self, isShow = False):
        frames = self.pipeline.wait_for_frames()
        ir_frame = frames.get_infrared_frame()
        ir_image = np.asanyarray(ir_frame.get_data())
        if isShow:
            cv2.imshow('RealSense: depth', ir_image)
            cv2.waitKey(0)
        #print("Under construction...")
        return ir_image

    def get_depth(self, x, y):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth = depth_frame.get_distance(x, y)
        return depth

    def stop(self):
        # Stop streaming
        self.Stop = True
        cv2.destroyAllWindows()

    def quit_rs(self):
        if self.isActive == True:
            self.pipeline.stop()
            self.isActive = False
        return

def main():
    rs = RealSenseL515()
    rs.start_rs()
    rs.display_cv()
    rs.stop()
    #depth_map = rs.get_depth_image(isShow=True)
    color_img = rs.get_color_image(isShow=True)

if __name__=="__main__":
    main()