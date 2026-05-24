import torch
import torch.nn as nn
import numpy as np

class BaseRenderModel(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)