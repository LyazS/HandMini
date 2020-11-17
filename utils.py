import numpy as np
import pickle

connections = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10),
               (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18),
               (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
reorder_mano = [
    0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18
]


def hmap2uv(hmap):
    """
    hmap:(1,32,32,21)
    """
    hmap_flat = hmap.reshape((1, -1, 21))
    argmax = np.argmax(hmap_flat, axis=1).astype(np.int)
    argmax_x = argmax // 32
    argmax_y = argmax % 32
    uv = np.stack((argmax_x, argmax_y), axis=1)
    uv = np.transpose(uv, [0, 2, 1])
    return uv[0]


def load_pkl(path):
    """
    Load pickle data.

    Parameter
    ---------
    path: Path to pickle file.

    Return
    ------
    Data in pickle file.

    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


class MANOHandJoints:
    n_joints = 21

    labels = [
        'W',  #0
        'I0',
        'I1',
        'I2',  #3
        'M0',
        'M1',
        'M2',  #6
        'L0',
        'L1',
        'L2',  #9
        'R0',
        'R1',
        'R2',  #12
        'T0',
        'T1',
        'T2',  #15
        'I3',
        'M3',
        'L3',
        'R3',
        'T3'  #20, tips are manually added (not in MANO)
    ]

    # finger tips are not joints in MANO, we label them on the mesh manually
    mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

    parents = [
        None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15
    ]


class MPIIHandJoints:
    n_joints = 21

    labels = [
        'W',  #0
        'T0',
        'T1',
        'T2',
        'T3',  #4
        'I0',
        'I1',
        'I2',
        'I3',  #8
        'M0',
        'M1',
        'M2',
        'M3',  #12
        'R0',
        'R1',
        'R2',
        'R3',  #16
        'L0',
        'L1',
        'L2',
        'L3',  #20
    ]

    parents = [
        None, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18,
        19
    ]


def mpii_to_mano(mpii):
    """
    Map data from MPIIHandJoints order to MANOHandJoints order.

    Parameters
    ----------
    mpii : np.ndarray, [21, ...]
    Data in MPIIHandJoints order. Note that the joints are along axis 0.

    Returns
    -------
    np.ndarray
    Data in MANOHandJoints order.
    """
    mano = []
    for j in range(MANOHandJoints.n_joints):
        mano.append(mpii[MPIIHandJoints.labels.index(
            MANOHandJoints.labels[j])])
    mano = np.stack(mano, 0)
    return mano


def mano_to_mpii(mano):
    """
    Map data from MANOHandJoints order to MPIIHandJoints order.

    Parameters
    ----------
    mano : np.ndarray, [21, ...]
    Data in MANOHandJoints order. Note that the joints are along axis 0.

    Returns
    -------
    np.ndarray
    Data in MPIIHandJoints order.
    """
    mpii = []
    for j in range(MPIIHandJoints.n_joints):
        mpii.append(mano[MANOHandJoints.labels.index(
            MPIIHandJoints.labels[j])])
    mpii = np.stack(mpii, 0)
    return mpii


def xyz_to_delta(xyz, joints_def):
    """
    Compute bone orientations from joint coordinates (child joint - parent joint).
    The returned vectors are normalized.
    For the root joint, it will be a zero vector.

    Parameters
    ----------
    xyz : np.ndarray, shape [J, 3]
    Joint coordinates.
    joints_def : object
    An object that defines the kinematic skeleton, e.g. MPIIHandJoints.

    Returns
    -------
    np.ndarray, shape [J, 3]
    The **unit** vectors from each child joint to its parent joint.
    For the root joint, it's are zero vector.
    np.ndarray, shape [J, 1]
    The length of each bone (from child joint to parent joint).
    For the root joint, it's zero.
    """
    delta = []
    for j in range(joints_def.n_joints):
        p = joints_def.parents[j]
        if p is None:
            delta.append(np.zeros(3))
        else:
            delta.append(xyz[j] - xyz[p])
    delta = np.stack(delta, 0)
    lengths = np.linalg.norm(delta, axis=-1, keepdims=True)
    delta /= np.maximum(lengths, np.finfo(xyz.dtype).eps)
    return delta, lengths


_MAX_FLOAT = np.maximum_sctype(np.float)
_FLOAT_EPS = np.finfo(np.float).eps


def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.

    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                     [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                     [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


import vpython as vp


def vpvec(arr):
    r"""
    将普通列表变量转换为python的vector
    """
    return vp.vector(arr[0], arr[1], arr[2])


class HandVis():
    def __init__(self, ):
        self.cav = vp.canvas()
        self.cav.camera.rotate(angle=vp.radians(180), axis=self.cav.up)
        self.cav.camera.rotate(angle=vp.radians(180), axis=self.cav.forward)
        joints_radius = 10
        joints_color = vpvec([1, 1, 1])
        joints_opacity = 0.8
        line_color = vp.color.cyan
        line_radius = 5

        self.joints = [
            vp.sphere(
                canvas=self.cav,
                pos=vpvec([0, 0, 0]),
                radius=joints_radius * 0.9,
                color=joints_color,
                opacity=joints_opacity,
            ) for i in range(21)
        ]
        self.jointline = {
            c: vp.curve(
                canvas=self.cav,
                pos=[vpvec([0,0,0]), vpvec([0,0,0])],
                color=line_color,
                radius=line_radius,
            )
            for c in connections
        }

    def visualize(self, xyz):
        xyz*=1000
        for j in range(21):
            self.joints[j].pos = vpvec(xyz[j])

        for c in connections:
            self.jointline[c].modify(0, vpvec(xyz[c[0]]))
            self.jointline[c].modify(1, vpvec(xyz[c[1]]))
