from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch
import numpy as np

@OPENOCC_LOSS.register_module()
class PlanRegLossLidar(BaseLoss):
    def __init__(self, weight=1.0, num_modes=3, input_dict=None, loss_type='l2', return_last=False, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'rel_pose': 'rel_pose',
                'metas': 'metas'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.plan_reg_loss
        self.num_mode = num_modes
        self.loss_type = loss_type
        self.return_last = return_last
        assert loss_type in ['l1', 'l2'], f'loss_type {loss_type} not supported'
        
    def plan_reg_loss(self, rel_pose, metas):
        bs, num_frames, num_modes, _ = rel_pose.shape
        
        rel_pose = rel_pose.transpose(1, 2) # B, M, F, 2
        rel_pose = torch.cumsum(rel_pose, -2)
        #print(rel_pose.shape)
        gt_rel_pose, gt_mode = [], []
        for meta in metas:
            gt_rel_pose.append(meta['rel_poses'])
            gt_mode.append(meta['gt_mode'])
        gt_rel_pose = rel_pose.new_tensor(np.asarray(gt_rel_pose)) # B, F, 2
        gt_mode = rel_pose.new_tensor(np.asarray(gt_mode)).transpose(1,2) # B, F, M ? maybe B, M, F
        gt_rel_pose = gt_rel_pose.unsqueeze(1).repeat(1, num_modes, 1, 1) # B, M, F, 2
        gt_rel_pose = torch.cumsum(gt_rel_pose, -2)
    
        if self.return_last:
            # rel_pose = rel_pose[:,:,-1:]
            # gt_rel_pose = gt_rel_pose[:,:,-1:]
            # gt_mode = gt_mode[:, -1:]
            # bs, num_modes, num_frames, _ = rel_pose.shape
            rel_pose = rel_pose.new_tensor(rel_pose[:, :, -1:])
            gt_mode = gt_mode.new_tensor(gt_mode[:, :, -1:])
            gt_rel_pose = gt_rel_pose.new_tensor(gt_rel_pose[:, :, -1:])
            bs, num_modes, num_frames, _ = rel_pose.shape
            assert num_frames == 1
            
        if self.loss_type == 'l1':
            weight = gt_mode[..., None].repeat(1, 1, 1, 2)
            loss = torch.abs(rel_pose - gt_rel_pose) * weight
        elif self.loss_type == 'l2':
            weight = gt_mode # [..., None].repeat(1, 1, 1)
            loss = torch.sqrt(((rel_pose - gt_rel_pose) ** 2).sum(-1)) * weight
        #loss = torch.abs(rel_pose - gt_rel_pose) * weight

        return loss.sum() / bs / num_frames