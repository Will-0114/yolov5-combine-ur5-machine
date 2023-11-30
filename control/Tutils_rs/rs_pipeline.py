## rs.pipeline
# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()
try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue
        depth_data = depth.as_frame().get_data()
        np_image = np.asanyarray(depth_data) #  np.asarray(depth_data, dtype = np.uint8)
        cv2.imshow("realsense depth 16 unit image (ESC stop)", np_image)
        heatmapshow = cv2.normalize(np_image, np_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #heatmapshow  = np_image/65535 #np.asarray(np_image/65535*255, dtype=np.uint8)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imshow("realsense depth image (ESC stop)", heatmapshow)
        key = cv2.waitKey(50)
        if key == 27:  # ESC the stop
            break
finally:
    pipeline.stop()