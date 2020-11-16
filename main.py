import numpy as np
import cv2
# import tensorflow as tf
import matplotlib.pyplot as plt
from utils import hmap2uv

connections = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10),
               (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18),
               (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

pb_path = "model/detnet.pb"

# with tf.gfile.FastGFile(pb_path, "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     inp, hmap_tf, lmap_tf = tf.import_graph_def(
#         graph_def,
#         return_elements=[
#             "prior_based_hand/input_0:0",
#             "prior_based_hand/hmap_0/prediction/conv2d/Sigmoid:0",
#             "prior_based_hand/Reshape:0",
#         ])

detnet = cv2.dnn.readNet(pb_path)

cv2.namedWindow("uv", cv2.WINDOW_NORMAL)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)
fig = plt.figure(0)
plt.ion()
ax = fig.gca(projection='3d')
while 1:
    _, frame = capture.read()
    if frame is None:
        break
    frame = frame.astype(np.float32) / 255
    size = min(frame.shape[:2])
    w_crop = (frame.shape[1] - size) // 2
    h_crop = (frame.shape[0] - size) // 2
    frame = frame[0 + h_crop:frame.shape[0] - h_crop,
                  0 + w_crop:frame.shape[1] - w_crop, :, ]
    frame = cv2.resize(frame, (128, 128), cv2.INTER_LINEAR)
    blob = np.transpose(frame, (2, 0, 1))[np.newaxis]
    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #     tfhmap, tflmap = sess.run([hmap_tf, lmap_tf],
    #                               feed_dict={inp: frame[np.newaxis]})

    detnet.setInput(blob)
    hmap, lmap = detnet.forward([
        "prior_based_hand/hmap_0/prediction/conv2d/Sigmoid",
        "prior_based_hand/Reshape",
    ])
    hmap=np.transpose(hmap, (0, 2, 3, 1))
    uv = hmap2uv(hmap)
    hmap_sum = np.sum(hmap[0], axis=-1)
    hmap_sum = (hmap_sum - np.min(hmap_sum)) * 255 / (np.max(hmap_sum) -
                                                      np.min(hmap_sum))
    hmap_sum = hmap_sum.astype(np.uint8)
    xyz = []
    for i, uvi in enumerate(uv):
        xyz.append(lmap[0, uvi[0], uvi[1], i, :])
        cv2.circle(frame, (uvi[1] * 4, uvi[0] * 4), 4, (0, 0, 255))
        # cv2.circle(frame, (tfuv[i, 1] * 4, tfuv[i, 0] * 4), 2, (255, 0, 0))
    xyz = np.stack(xyz, axis=0)
    cv2.imshow("uv", hmap_sum)
    cv2.imshow("img", frame)
    cv2.waitKey(1)

    ax.cla()
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="b")
    for c in connections:
        ax.plot(xyz[c, 0], xyz[c, 1], xyz[c, 2], c="b")
    # ax.scatter(tfxyz[:, 0], tfxyz[:, 1], tfxyz[:, 2], c="r")
    # for c in connections:
    #     ax.plot(tfxyz[c, 0], tfxyz[c, 1], tfxyz[c, 2], c="r")
    plt.pause(0.0001)
