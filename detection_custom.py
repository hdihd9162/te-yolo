
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image, detect_video, detect_realtime
from yolov3.configs import *


 

image_path   = "./IMAGES/plate_2.jpg"
video_path   = "./IMAGES/test.mp4"


yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE)
yolo.load_weights("./checkpoints/yolov3_custom") 
    
#detect_image(yolo, image_path, "./IMAGES/plate_1_detect.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
