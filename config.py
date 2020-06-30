# -------------------
# Written by Xusheng Li
# -------------------
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
    def __init__(self):
        self.IMG_FOLDER = ''
        self.MODEL_DEF = 'config/yolov3-custom.cfg'
        self.WEIGHTS_PATH = 'checkpoints/yolov3_ckpt_99.pth'
        self.CLASS_PATH = 'data/custom/classes.names'
# conf_thres and nms_thres can be tuned for performance
        self.CONF_THRES = 0.8
        self.NMS_THRES = 0.4
        self.BATCH_SIZE = 1
        self.IMG_SIZE = 416
        self.CKPT_MODEL = 'checkpoints/yolov3_ckpt_99.pth'
# video_path should be changed if wishing to test another video
        self.VIDEO_PATH = 'examples/视频2红外.mp4'
        self.SAVE_PATH = 'examples/res'

opt = Configuration()
