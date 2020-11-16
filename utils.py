import numpy as np

def hmap2uv(hmap):
    """
    hmap:(1,32,32,21)
    """
    hmap_flat = hmap.reshape( (1, -1, 21))
    argmax = np.argmax(hmap_flat, axis=1).astype(np.int)
    argmax_x = argmax // 32
    argmax_y = argmax % 32
    uv = np.stack((argmax_x, argmax_y), axis=1)
    uv = np.transpose(uv, [0, 2, 1])
    return uv[0]