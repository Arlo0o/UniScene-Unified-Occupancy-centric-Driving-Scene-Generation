from .base_loss import BaseLoss
from . import OPENOCC_LOSS

@OPENOCC_LOSS.register_module()
class VQVAEEmbedLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'embed_loss': 'embed_loss'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.embed_loss
    
    def embed_loss(self, embed_loss):
        return embed_loss

@OPENOCC_LOSS.register_module()
class KL_Loss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'kl_loss': 'kl_loss'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.KL_loss
    
    def KL_loss(self, kl_loss):
        return kl_loss