from .detector3d_template import Detector3DTemplate
from ..backbones_3d.vfe.dynamic_frustum_vfe import DynamicFrustumVFE
from .. import dense_heads

class Occ2LiDARSparseUNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # vfe (no need)
        if isinstance(self.module_list[0], DynamicFrustumVFE):
           # frumstum voxelize (theta, phi, r)
           batch_dict = self.module_list[0]({'points': batch_dict['occ'][:, [0, 8, 9, 10]]})
        else:
           batch_dict = self.module_list[0](batch_dict)
        
        batch_dict['gt_voxel_features'] = batch_dict['voxel_features']
        batch_dict['gt_voxel_coords'] = batch_dict['voxel_coords']
        
        if self.model_cfg.get('USE_CLS', True):
            batch_dict['voxel_features'] = batch_dict['occ'][:, 5:]
        else:
            batch_dict['voxel_features'] = batch_dict['occ'][:, 5:11]
        batch_dict['voxel_coords'] = batch_dict['occ'][:, 0:4]

        for cur_module in self.module_list[1:]:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts = self.post_processing(batch_dict)
            return pred_dicts, None

    def get_training_loss(self):
        disp_dict = {}

        loss_head, tb_dict = self.dense_head.get_loss()

        loss = loss_head
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        return self.dense_head.generate_predicted_pc(batch_dict)
    

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD.get('INPUT_FEATURES', None),
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.get('CLASS_AGNOSTIC', None) else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict
