#!/usr/bin/python3

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import freenect
import numpy as np
import tensorflow as tf
from object_detection.utils.label_map_util import \
    create_category_index_from_labelmap as create_category_index
from object_detection.utils.visualization_utils import \
    visualize_boxes_and_labels_on_image_array as vis
from scipy.spatial import distance

import frame_convert
from utils import debounce

model_name = 'ssd_mobilenet_v3_large_coco_2020_01_14'
origin = f'http://download.tensorflow.org/models/object_detection/{model_name}.tar.gz'
model_dir = tf.keras.utils.get_file(fname=model_name, origin=origin, untar=True)

interpreter = tf.lite.Interpreter(os.path.join(model_dir, 'model.tflite'))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

category_index = create_category_index('mscoco_label_map.pbtxt', use_display_name=True)
category_index = {v['id'] - 1: v for v in category_index.values()}

find = 'bottle'
id = [k for k, v in category_index.items() if str(v['name']).lower() == find][0]

pos = (0, 0, 0)
last_pos = (0, 0, 0)


@debounce(1)
def write_pos():
    print(pos)


while True:

    frame = frame_convert.video_cv(freenect.sync_get_video()[0])
    input_tensor = tf.convert_to_tensor(np.expand_dims(cv2.resize(frame, (320, 320)), 0))

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    _boxes = interpreter.get_tensor(output_details[0]['index'])
    _classes = interpreter.get_tensor(output_details[1]['index'])
    _scores = interpreter.get_tensor(output_details[2]['index'])

    boxes = classes = scores = []

    try:
        boxes, classes, scores = zip(*((b, c, s)
                                       for b, c, s in zip(_boxes[0], _classes[0], _scores[0])
                                       if c == id))

        if len(boxes) > 1:
            x = np.argmax(scores)
            boxes = [boxes[x]]
            classes = [classes[x]]
            scores = [scores[x]]
    except:
        pass

    boxes = np.array(boxes)
    classes = np.array(classes, dtype=int)
    scores = np.array(scores)
    vis(frame, boxes, classes, scores, category_index, use_normalized_coordinates=True)

    cv2.imshow('perception', frame)
    if cv2.waitKey(10) == 27:
        break

    if len(scores) and scores[0] > 0.5:
        [y_min, x_min, y_max, x_max] = boxes[0]
        (left, right, top, bottom) = (x_min * 640, x_max * 640, y_min * 480, y_max * 480)

        depth = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)[0]

        xyz = freenect.depth_to_xy(depth)
        x_arr = []
        y_arr = []
        z_arr = []

        for py in range(int(top), int(bottom)):
            for px in range(int(left), int(right)):
                if not (0 <= px < 640 and 0 <= py < 480):
                    continue

                x, y, z = xyz[py][px]

                if 300 < z < 1000:
                    x_arr.append(x)
                    y_arr.append(y)
                    z_arr.append(z)

        if len(x_arr):
            x = np.median(x_arr) // 10
            y = np.median(y_arr) // 10
            z = np.median(z_arr) // 10

            if distance.euclidean(pos, (x, y, z)) > 1:
                pos = (x, y, z)
                write_pos()
