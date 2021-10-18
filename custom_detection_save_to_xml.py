# ================================================================
#
#   File name   : detection_save_to_xml.py
#   Author      : Istvan Illes-Toth
#   Created date: 2021-10-16
#   Website     : https://xenial.com/
#   GitHub      : https://github.com/illestothi/TensorFlow-2.x-YOLOv3
#   Description : DO NOT FORGET CONFIGS.PY YOLO_CUSTOM_WEIGHTS = True
#
# ================================================================
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

train_path = "./IMAGES/car/train/images"
validation_path = "./IMAGES/car/validation/images"
train_annotations_path = "./IMAGES/car/train/annotations"
validation_annotations_path = "./IMAGES/car/validation/annotations"


def create_annotation_xml(filename, content):
    with open(train_annotations_path + '/' + filename, "w") as f:
        f.truncate()
        f.write(content)
        f.close()


def read_annotation_xml(name):
    content = ""
    with open(name) as f:
        for line in f:
            content += line
        f.close()

    return content


yolo = Load_Yolo_model()
object_xml = read_annotation_xml('./xml/object.xml')

i = 0
for filename in os.listdir(train_path):
    i += 1
    if i > 2:
        pass
    image_path = os.path.join(train_path, filename)
    annotation_path = os.path.join(train_annotations_path, filename.replace('jpg', 'xml'))
    if os.path.isfile(annotation_path):
        skeleton_xml = read_annotation_xml(annotation_path)
    else:
        # WE ASSUME ANNOTATION FILE IS EXIST BEFORE YOU RUN
        # THIS IS NECESSARY BECAUSE WE MUST RUN FIRST THE DETECTION_SAVE_TO_XML.PY
        print('ERROR', 'ANNOTATION FILE DOES NOT EXISTS: ', annotation_path)
        continue

    detections = detect_image(yolo, image_path, "./IMAGES/car_res.jpg", input_size=YOLO_INPUT_SIZE,
                              CLASSES=TRAIN_CLASSES, show=False, rectangle_colors=(255, 0, 0),
                              score_threshold=0.6)

    if detections is not None:
        objects_xml_content = ""
        for detection in detections:
            class_name = "unknown"
            x1, y1, x2, y2, probability, class_id = detection
            x1 = round(x1)
            x2 = round(x2)
            y1 = round(y1)
            y2 = round(y2)
            class_id = int(class_id)
            if class_id == 0:
                class_name = 'wheel'
                print('FOUND:', class_name, x1, y1, x2, y2, probability, int(class_id))
                object = object_xml.replace('[CLASS_NAME]', class_name)
                object = object.replace('[X1]', str(x1))
                object = object.replace('[X2]', str(x2))
                object = object.replace('[Y1]', str(y1))
                object = object.replace('[Y2]', str(y2))
                objects_xml_content += object  # + '\n'
        if objects_xml_content != "":
            xml_content = skeleton_xml.replace('</object>', str('</object>\n' + objects_xml_content), 1)
            xml_content = xml_content.replace('<path>./IMAGES/car/validation/images</path>',
                                              str('<path>' + str(image_path) + '</path>'))
            create_annotation_xml(filename.replace('jpg', 'xml'), xml_content)
            print(filename)
