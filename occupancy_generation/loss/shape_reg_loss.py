from .base_loss import BaseLoss
from . import OPENOCC_LOSS

@OPENOCC_LOSS.register_module()
class ShapeRegLoss(BaseLoss):
    """
    形状正则化损失函数
    用于约束强制方形变换的平滑性
    """

    def __init__(self, weight=0.005, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'shape_reg_loss': 'shape_reg_loss'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.shape_reg_loss
    
    def shape_reg_loss(self, shape_reg_loss):
        """
        计算形状正则化损失
        
        Args:
            shape_reg_loss: 从模型输出中获取的形状正则化损失
            
        Returns:
            torch.Tensor: 形状正则化损失值
        """
        return shape_reg_loss
