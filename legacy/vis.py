import math 
import time
import cv2
import numpy as np
import torch
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 255)
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def vis_frame(frame, im_res, thres=0.05, show_conf=True):
    """
        ims_res: (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    img = frame.copy()
    height, width = img.shape[:2]
    for bbox in im_res:
        x1, y1, x2, y2, object_conf, class_score, class_pred = bbox
        if object_conf > thres:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),  GREEN, 2)
            if show_conf:
                cv2.putText(img, ''.join(str(object_conf)), (int(x1), int((y1)+26)), DEFAULT_FONT, 1, GREEN, 2)
    return img

 
