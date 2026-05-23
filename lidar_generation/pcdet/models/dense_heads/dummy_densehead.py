from torch import nn

class DummyDenseHead(nn.Module):
    def __init__(
        self,
        model_cfg,
        **kwargs
    ):
        super().__init__()
        self.forward_ret_dict = {}

    def get_loss(self):
        
        return self.forward_ret_dict['feats'].sum(), {}
    
        
    def forward(self, batch_dict):#pts_feats, rays):
        self.forward_ret_dict['feats'] = batch_dict['point_features']
        return batch_dict
