import torch.nn as nn
from mmengine.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class PoseEncoder(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1
        
        pose_encoder = []

        for _ in range(num_layers - 1):
            pose_encoder.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU(True)])
            in_channels = out_channels
        pose_encoder.append(nn.Linear(out_channels, out_channels))
        self.pose_encoder = nn.Sequential(*pose_encoder)
    
    def forward(self,x):
        # x: N*2,
        pose_feat = self.pose_encoder(x)
        return pose_feat
        