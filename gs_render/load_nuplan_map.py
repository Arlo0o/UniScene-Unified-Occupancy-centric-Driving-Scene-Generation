import numpy as np



def one_hot_decode(data: np.ndarray, n: int):
    """
    returns (h, w, n) np.int64 {0, 1}
    """
    # shift = np.arange(n, dtype=np.int32)[None, None]
    shift = np.zeros((1, 1, n), np.int32)
    shift[0, 0, :] = np.arange(0, n, 1, np.int32)

    # x = np.array(data)[..., None]
    x = np.zeros((*data.shape, 1), data.dtype)
    x[..., 0] = data
    # after shift, numpy keeps int32, numba changes dtype to int64
    x = (x >> shift) & 1  # only keep the lowest bit, for each n

    x = x.transpose(2, 0, 1)
    return x

def load_occ_layout_nuplan(layout_path):
    bevmap = np.load(layout_path)
    bevmap = one_hot_decode(bevmap, 18)
    return bevmap
