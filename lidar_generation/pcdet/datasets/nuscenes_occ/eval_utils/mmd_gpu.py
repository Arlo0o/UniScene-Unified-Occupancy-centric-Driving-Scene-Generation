import concurrent.futures
import pickle
from functools import partial
from typing import Any, Dict, Sequence
from tqdm import tqdm
import numpy as np
from numba import njit, prange
import torch
from torch.nn import functional as F
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

__all__ = ["MaximumMeanDiscrepancy"]

def debug_viz(s1, s2):
    import matplotlib.pyplot as plt 
    f, axarr = plt.subplots(2)
    axarr[0].imshow(s1)
    axarr[1].imshow(s2)
    f.show()


class MaximumMeanDiscrepancy(Metric):
    def __init__(self) -> None:
        super().__init__(dist_sync_on_step=False)

        self.add_state("gt_set", default=[], dist_reduce_fx="cat")
        self.add_state("gen_set", default=[], dist_reduce_fx="cat")

    def update(self, data: Dict[str, Any]) -> None:
        #debug_viz(self._flatten_samples(data["lidar"]).sum(0), self._flatten_samples(data["sample"]).sum(0))

        self.gen_set.append(self._flatten_samples(data["sample"]))
        self.gt_set.append(self._flatten_samples(data["lidar"]))

    def _flatten_samples(self, sample):
        return F.adaptive_avg_pool3d(sample, (1, 100, 100)).squeeze(1)

    def compute(self, middle_save_path=None):
        catted1 = dim_zero_cat(self.gt_set)
        dist1_unnormalized = catted1.float()#.cpu().numpy()
        batch_size1 = dist1_unnormalized.shape[0]
        dist1_unnormalized_list = [dist1_unnormalized[i].flatten() for i in range(batch_size1)]

        catted2 = dim_zero_cat(self.gen_set)
        dist2_unnormalized = catted2.float()#.cpu().numpy()
        batch_size2 = dist2_unnormalized.shape[0]
        dist2_unnormalized_list = [dist2_unnormalized[i].flatten() for i in range(batch_size2)]

        mmd = compute_mmd(dist1_unnormalized_list, dist2_unnormalized_list, gaussian, is_hist=True, middle_save_path=middle_save_path)
        return mmd

def gaussian(x, y, sigma=0.5):
    support_size = max(len(x), len(y))

    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    # TODO: Calculate empirical sigma by fitting dist to gaussian
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))

@njit
def gaussian_jit(x, y, sigma=0.5):
    support_size = max(len(x), len(y))

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, np.zeros((support_size - len(x)))))
    elif len(y) < len(x):
        y = np.hstack((y, np.zeros((support_size - len(y)))))

    # TODO: Calculate empirical sigma by fitting dist to gaussian
    # dist = np.linalg.norm(x - y, 2)
    dist = 0.0
    for i in range(support_size):
        dist += (x[i] - y[i]) ** 2
    dist = np.sqrt(dist)

    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)

@njit(parallel=True)
def disc_jit(samples1, samples2):
    d = 0.0
    for i in prange(len(samples1)):
        for j in prange(len(samples2)):
            d += gaussian_jit(samples1[i], samples2[j])
    d /= len(samples1) * len(samples2)
    return d

def disc_gpu(samples1, samples2):
    sigma=0.5

    samples1 = samples1.to(torch.float64)
    samples2 = samples2.to(torch.float64)

    res = 0.0
    N = samples1.size(0)
    B = 800  # 您可以根据显存大小调整块的大小

    for i in range(0, N, B):
        samples1_block = samples1[i:i+B]  # [B, C]
        for j in range(0, N, B):
            samples2_block = samples2[j:j+B]  # [B, C]
            dist = torch.linalg.norm(
                samples1_block.unsqueeze(1) - samples2_block.unsqueeze(0),
                dim=-1
            )
            res_block = torch.exp(-dist * dist / (2 * sigma * sigma))
            res += res_block.sum()

    # dist = torch.linalg.norm(samples1.unsqueeze(1) - samples2.unsqueeze(0), dim=-1)
    # res = torch.exp(-dist * dist / (2 * sigma * sigma))
    # res = res.sum()

    res /= (samples1.shape[0] * samples2.shape[0])
    return res.item()

def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0

    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dist in tqdm(executor.map(
                kernel_parallel_worker, [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]
            )):
                d += dist

    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, middle_save_path=None, *args, **kwargs):
    """MMD between two samples"""
    if is_hist:
        samples1 = [s1 / (torch.sum(s1)+1e-6) for s1 in samples1]
        samples2 = [s2 / (torch.sum(s2)+1e-6) for s2 in samples2]
    # return (
    #     disc(samples1, samples1, kernel, *args, **kwargs)
    #     + disc(samples2, samples2, kernel, *args, **kwargs)
    #     - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    # )

    if middle_save_path is not None:
        with open(middle_save_path, 'wb') as f:
            pickle.dump([samples1, samples2], f)
        print(f'Save MMD input to {middle_save_path}')

    # (N, C)
    samples1 = torch.stack(samples1).cuda()
    samples2 = torch.stack(samples2).cuda()
    return (
        disc_gpu(samples1, samples1)
        + disc_gpu(samples2, samples2)
        - 2 * disc_gpu(samples1, samples2)
    )
