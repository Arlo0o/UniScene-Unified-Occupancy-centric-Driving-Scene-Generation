from .detector3d_template import Detector3DTemplate
from .occ2lidar_sparseunet import Occ2LiDARSparseUNet
from .occ2lidar_ptv3 import Occ2LiDARPTv3

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'Occ2LiDARSparseUNet': Occ2LiDARSparseUNet,
    'Occ2LiDARPTv3': Occ2LiDARPTv3
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
