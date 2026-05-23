from typing import Any, Dict, Sequence

import torch
from torchmetrics import Metric

__all__ = ["BinaryJaccardIndex"]


class BinaryJaccardIndex(Metric):
    def __init__(self, output_key: str, target_key: str, thresholds: Sequence[float]) -> None:
        super().__init__(dist_sync_on_step=False)

        self.output_key = output_key
        self.target_key = target_key

        self.register_buffer("thresholds", torch.tensor(thresholds))

        self.add_state("i", default=torch.zeros(len(thresholds)), dist_reduce_fx="sum")
        self.add_state("u", default=torch.zeros(len(thresholds)), dist_reduce_fx="sum")

    def update(self, data: Dict[str, Any]) -> None:
        output = data[self.output_key].reshape(-1).sigmoid()
        target = data[self.target_key].reshape(-1)

        output = output[..., None] > self.thresholds
        target = target[..., None] > self.thresholds

        self.i += torch.sum(output & target, dim=0)
        self.u += torch.sum(output | target, dim=0)

    def compute(self) -> torch.Tensor:
        return torch.max(self.i / self.u.clamp(min=1e-5))
