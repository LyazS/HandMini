# python -m tf2onnx.convert --graphdef iknet.pb --output iknet.onnx --inputs Placeholder:0 --outputs network/Reshape_1:0
# python -m tf2onnx.convert --graphdef detnet.pb --output detnet.onnx --inputs prior_based_hand/input_0:0 --outputs prior_based_hand/hmap_0/prediction/conv2d/Sigmoid:0,prior_based_hand/Reshape:0

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from detiknet import Pipeline

detiknet = Pipeline("model/detnet.onnx", "model/iknet.onnx")
cv2.namedWindow("uv", cv2.WINDOW_NORMAL)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)

handvis = HandVis()
while 1:
    _, frame = capture.read()
    if frame is None:
        break
    frame = np.flip(frame, 1)  # 反转以只检测右手
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255
    size = min(frame.shape[:2])
    w_crop = (frame.shape[1] - size) // 2
    h_crop = (frame.shape[0] - size) // 2
    frame = frame[0 + h_crop:frame.shape[0] - h_crop, 0 +
                  w_crop:frame.shape[1] - w_crop, :, ]
    frame = cv2.resize(frame, (128, 128), cv2.INTER_LINEAR)
    blob = frame[np.newaxis]

    starttime = time.time()
    uv, xyz, xyz_mano, vert_mano = detiknet(blob)
    print(time.time() - starttime)

    for c in connections:
        cv2.line(
            frame,
            (uv[c[0], 1] * 4, uv[c[0], 0] * 4),
            (uv[c[1], 1] * 4, uv[c[1], 0] * 4),
            color=(0, 0, 255),
            thickness=3,
        )
    cv2.imshow("img", frame)
    cv2.waitKey(1)

    handvis.visualize(xyz_mano)
