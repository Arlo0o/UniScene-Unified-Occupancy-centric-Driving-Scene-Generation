from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .ce_loss import CeLoss
from .plan_reg_loss_lidar import PlanRegLossLidar
from .emb_loss import VQVAEEmbedLoss, KL_Loss
from .recon_loss import ReconLoss, LovaszLoss
from .shape_reg_loss import ShapeRegLoss