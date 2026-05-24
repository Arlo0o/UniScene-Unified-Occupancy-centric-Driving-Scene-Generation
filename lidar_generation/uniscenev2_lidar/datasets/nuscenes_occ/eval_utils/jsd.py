from typing import Any, Dict, Tuple

import pickle
import torch
from scipy.spatial.distance import jensenshannon
from torch.nn import functional as F
from torchmetrics import Metric

__all__ = ["JensenShannonDivergence"]


# def jensenshannon(p, q):
#     p = p / p.sum()
#     q = q / q.sum()

#     m = 0.5 * (p + q)
#     return 0.5 * (F.kl_div(p.log(), m) + F.kl_div(q.log(), m))


def resize(x, size):
    return F.adaptive_avg_pool3d(x, size)


class JensenShannonDivergence(Metric):
    def __init__(self, spatial_shape: Tuple[int, int, int]) -> None:
        super().__init__(dist_sync_on_step=False)

        self.spatial_shape = spatial_shape

        self.add_state("p", default=torch.zeros(*spatial_shape), dist_reduce_fx="sum")
        self.add_state("q", default=torch.zeros(*spatial_shape), dist_reduce_fx="sum")

    def update(self, data: Dict[str, Any]) -> None:
        self.p += resize(data["sample"], self.spatial_shape).sum(dim=0)
        self.q += resize(data["lidar"], self.spatial_shape).sum(dim=0)

    def compute(self, middle_save_path=None) -> torch.Tensor:
        p = self.p.cpu().numpy().flatten()
        q = self.q.cpu().numpy().flatten()

        if middle_save_path is not None:
            with open(middle_save_path, 'wb') as f:
                pickle.dump([p, q], f)
                print(f'Save JSD input to {middle_save_path}')
        return jensenshannon(p, q)
