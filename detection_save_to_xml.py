# ================================================================
#
#   File name   : detection_save_to_xml.py
#   Author      : Istvan Illes-Toth
#   Created date: 2021-10-16
#   Website     : https://xenial.com/
#   GitHub      : https://github.com/illestothi/TensorFlow-2.x-YOLOv3
#   Description :
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
    with open(validation_annotations_path + '/' + filename, "a") as f:
        f.write(content)
        f.close()


def read_skeleton_xml(name='skeleton.xml'):
    content = ""
    with open("./xml/" + name) as f:
        for line in f:
            content += line
        f.close()

    return content


yolo = Load_Yolo_model()
skeleton_xml = read_skeleton_xml()
object_xml = read_skeleton_xml('object.xml')
# print("object_xml:", object_xml)
i = 0
for filename in os.listdir(validation_path):
    i += 1
    if i > 10:
        pass
        # break
    image_path = os.path.join(validation_path, filename)
    annotation_path = os.path.join(validation_annotations_path, filename.replace('jpg', 'xml'))
    if os.path.isfile(annotation_path):
        continue  # if annotation is already exist we skip this img

    detections = detect_image(yolo, image_path, "./IMAGES/car_res.jpg", input_size=YOLO_INPUT_SIZE,
                              CLASSES="model_data/coco/coco.names", show=False, rectangle_colors=(255, 0, 0),
                              score_threshold=0.85)

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
            if class_id in (2, 7):
                if class_id == 2:
                    class_name = 'car'
                if class_id == 7:
                    class_name = 'truck'
                print('FOUND:', class_name, x1, y1, x2, y2, probability, int(class_id))
                object = object_xml.replace('[CLASS_NAME]', class_name)
                object = object.replace('[X1]', str(x1))
                object = object.replace('[X2]', str(x2))
                object = object.replace('[Y1]', str(y1))
                object = object.replace('[Y2]', str(y2))
                objects_xml_content += object + '\n'
        if objects_xml_content != "":
            xml_content = skeleton_xml.replace('[FOLDER]', 'images')
            xml_content = xml_content.replace('[FILENAME]', str(filename))
            xml_content = xml_content.replace('[IMG_PATH]', str(image_path))
            img = cv2.imread(image_path)
            w, h, c = img.shape
            img = None
            xml_content = xml_content.replace('[WIDTH]', str(w))
            xml_content = xml_content.replace('[HEIGHT]', str(h))
            xml_content = xml_content.replace('[CHANNEL]', str(c))
            xml_content = xml_content.replace('[OBJECTS]', str(objects_xml_content))
            create_annotation_xml(filename.replace('jpg', 'xml'), xml_content)
