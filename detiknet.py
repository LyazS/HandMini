import onnxruntime as ort
import numpy as np
import onnx
import numpy as np
from utils import *


class IKNet_onnx():
    def __init__(self, iknet_path="model/iknet.onnx"):
        # so = ort.SessionOptions()
        # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # so.intra_op_num_threads = 1
        # so.inter_op_num_threads = 1
        self.ort_sess = ort.InferenceSession(iknet_path, sess_options=None)
        # self.ort_sess.set_providers(["CPUExecutionProvider"])

        self.in_shape = self.ort_sess.get_inputs()[0].shape
        self.out_shape = self.ort_sess.get_outputs()[0].shape

    def __call__(self, x):
        onnx_out = self.ort_sess.run(None, {'Placeholder:0': x})
        return onnx_out[0]


class DetNet_onnx():
    def __init__(self, detnet_path="model/detnet.onnx"):
        # so = ort.SessionOptions()
        # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # so.intra_op_num_threads = 1
        # so.inter_op_num_threads = 1
        self.ort_sess = ort.InferenceSession(detnet_path, sess_options=None)
        # self.ort_sess.set_providers(["CPUExecutionProvider"])

        self.in_shape = self.ort_sess.get_inputs()[0].shape
        self.out_shape = self.ort_sess.get_outputs()[0].shape

    def __call__(self, x):
        onnx_out = self.ort_sess.run(None, {"prior_based_hand/input_0:0": x})
        return onnx_out


class Pipeline():
    def __init__(self, detnet_path, iknet_path):
        self.detnet = DetNet_onnx(detnet_path)
        self.iknet = IKNet_onnx(iknet_path)

        HAND_MESH_MODEL_PATH = 'model/hand_mesh/hand_mesh_model.pkl'
        # use left hand
        OFFICIAL_MANO_PATH = 'model/mano_v1_2/models/MANO_LEFT.pkl'
        IK_UNIT_LENGTH = 0.09473151311686484  # in meter

        # load reference MANO hand pose
        params = load_pkl(HAND_MESH_MODEL_PATH)
        mano_ref_xyz = params['joints']

        self.verts = params['verts']
        self.faces = params['faces']
        self.weights = params['weights']
        self.joints = params['joints']

        self.n_verts = self.verts.shape[0]
        self.n_faces = self.faces.shape[0]

        self.ref_pose = []
        self.ref_T = []
        for j in range(MANOHandJoints.n_joints):
            parent = MANOHandJoints.parents[j]
            if parent is None:
                self.ref_T.append(self.verts)
                self.ref_pose.append(self.joints[j])
            else:
                self.ref_T.append(self.verts - self.joints[parent])
                self.ref_pose.append(self.joints[j] - self.joints[parent])
        self.ref_pose = np.expand_dims(np.stack(self.ref_pose, 0), -1)
        self.ref_T = np.expand_dims(np.stack(self.ref_T, 1), -1)

        # convert the kinematic definition to MPII style, and normalize it
        mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
        mpii_ref_xyz -= mpii_ref_xyz[9:10]
        # get bone orientations in the reference pose
        mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz,
                                                       MPIIHandJoints)
        mpii_ref_delta = mpii_ref_delta * mpii_ref_length

        self.mpii_ref_xyz = mpii_ref_xyz
        self.mpii_ref_delta = mpii_ref_delta

    def set_abs_quat(self, quat):
        """
        Set absolute (global) rotation for the hand.

        Parameters
        ----------
        quat : np.ndarray, shape [J, 4]
        Absolute rotations for each joint in quaternion.

        Returns
        -------
        np.ndarray, shape [V, 3]
        Mesh vertices after posing.
        """
        mats = []
        for j in range(MANOHandJoints.n_joints):
            mats.append(quat2mat(quat[j]))
        mats = np.stack(mats, 0)

        pose = np.matmul(mats, self.ref_pose)
        joint_xyz = [None] * MANOHandJoints.n_joints
        for j in range(MANOHandJoints.n_joints):
            joint_xyz[j] = pose[j]
            parent = MANOHandJoints.parents[j]
            if parent is not None:
                joint_xyz[j] += joint_xyz[parent]
        joint_xyz = np.stack(joint_xyz, 0)[..., 0]

        T = np.matmul(np.expand_dims(mats, 0), self.ref_T)[..., 0]
        self.verts = [None] * MANOHandJoints.n_joints
        for j in range(MANOHandJoints.n_joints):
            self.verts[j] = T[:, j]
            parent = MANOHandJoints.parents[j]
            if parent is not None:
                self.verts[j] += joint_xyz[parent]
        self.verts = np.stack(self.verts, 1)
        self.verts = np.sum(self.verts * self.weights, 1)

        return self.verts.copy(), joint_xyz

    def __call__(self, blob):
        hmap, lmap = self.detnet(blob)
        uv = hmap2uv(hmap)
        xyz = []
        for i, uvi in enumerate(uv):
            xyz.append(lmap[0, uvi[0], uvi[1], i, :])
        xyz = np.stack(xyz, 0)
        delta, length = xyz_to_delta(xyz, MPIIHandJoints)
        delta *= length
        pack = np.concatenate(
            [xyz, delta, self.mpii_ref_xyz, self.mpii_ref_delta],
            0).astype(np.float32)
        theta_raw = self.iknet(pack[np.newaxis])
        eps = np.finfo(np.float32).eps
        norm = np.maximum(np.linalg.norm(theta_raw, axis=-1, keepdims=True),
                          eps)
        theta_pos = theta_raw / norm
        theta_neg = theta_pos * -1
        theta_mpii = np.where(np.tile(theta_pos[:, :, 0:1] > 0, [1, 1, 4]),
                              theta_pos, theta_neg)
        theta_mano = mpii_to_mano(theta_mpii[0])
        vert_mano, xyz_mano = self.set_abs_quat(theta_mano)
        xyz_mano=xyz_mano[reorder_mano]
        return uv, xyz, xyz_mano, vert_mano


# ik = IKNet_onnx()

# for i in range(1000):
#     x = np.random.rand(1, 84, 3).astype(np.float32)
#     y = ik(x)
# print(y.shape)

# det = DetNet_onnx()
# x = np.random.rand(1, 128, 128, 3).astype(np.float32)
# y = det(x)
# print(y[0].shape, y[1].shape)

# pb_path = "model/detnet.pb"

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

# detnet = cv2.dnn.readNet(pb_path)

# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     tfhmap, tflmap = sess.run([hmap_tf, lmap_tf],
#                               feed_dict={inp: frame[np.newaxis]})

# detnet.setInput(blob)
# hmap, lmap = detnet.forward([
#     "prior_based_hand/hmap_0/prediction/conv2d/Sigmoid",
#     "prior_based_hand/Reshape",
# ])
# hmap = np.transpose(hmap, (0, 3, 1, 2))