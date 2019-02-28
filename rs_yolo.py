# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:53:24 2019

@author: lebl1803
"""
import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
print("Environment Ready")

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
#cfg.enable_device_from_file("../object_detection.bag")
profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
  pipe.wait_for_frames()
  
# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

# Cleanup:
pipe.stop()
print("Frames Captured")

color = np.asanyarray(color_frame.get_data())
plt.rcParams["axes.grid"] = False
plt.rcParams['figure.figsize'] = [12, 6]
plt.imshow(color)

#https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
#https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

#height, width = color.shape[:2]
#expected = 300
#aspect = width / height
#resized_image = cv2.resize(color, (round(expected * aspect), expected))
#crop_start = round(expected * (aspect - 1) / 2)
#crop_img = resized_image[0:expected, crop_start:crop_start+expected]

plt.savefig('image1.jpg')

import subprocess
subprocess.call(["python","yolo_opencv.py","--image","image1.jpg","--config","yolov3.cfg","--weights","yolov3.weights","--classes","yolov3.txt"])
