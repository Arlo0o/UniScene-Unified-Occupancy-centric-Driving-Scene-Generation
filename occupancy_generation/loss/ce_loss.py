from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F
import torch

@OPENOCC_LOSS.register_module()
class CeLoss(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
    
    def ce_loss(self, ce_inputs, ce_labels):
        # input: -1, c
        # output: -1, 1
        ce_loss = F.cross_entropy(ce_inputs, ce_labels)
        return ce_loss