import numpy as np, torch
from torch.utils import data
import torch.nn.functional as F
from copy import deepcopy
from mmengine import MMLogger
logger = MMLogger.get_instance('genocc')
from . import OPENOCC_DATAWRAPPER


@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):

        input, target, metas= self.point_cloud_dataset[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        if self.phase == 'train':
            if np.random.rand(1)<=0.5:
                input = torch.flip(input, dims = [2])
                target = torch.flip(target ,dims= [2])

        return input, target, metas


@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes_step2(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):

        input, target, metas, bevmaps = self.point_cloud_dataset[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        # bevmaps = torch.from_numpy(bevmaps) 
        return input, target, metas, bevmaps

@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes_HR_woBev(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):

        input, target, metas, bevmaps = self.point_cloud_dataset[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        # bevmaps = torch.from_numpy(bevmaps) 
        return input, target, metas#, bevmaps


@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes_HR_Bev(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):

        input, target, metas, bevmaps = self.point_cloud_dataset[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        bevmaps = torch.from_numpy(bevmaps) 
        return input, target, metas, bevmaps

def custom_collate_fn_temporal(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, torch.Tensor):
            data_tuple.append(torch.stack([d[i] for d in data]))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple


@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes_step2_wobev(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):

        input, target, metas, bevmaps = self.point_cloud_dataset[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        bevmaps = None
        # bevmaps = torch.from_numpy(bevmaps) 
        return input, target, metas #, bevmaps

@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuplan_pro_occ_bev(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):

        input, target, metas, bev_maps, aux_datas= self.point_cloud_dataset[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        if self.phase == 'train':
            if np.random.rand(1)<=0.5:
                input = torch.flip(input, dims = [2])
                target = torch.flip(target ,dims= [2])

        return input, target, metas



@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuplan_hr_mini(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):

        input, target, bev_maps, metas, aux_datas= self.point_cloud_dataset[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        if self.phase == 'train':
            if np.random.rand(1)<=0.5:
                input = torch.flip(input, dims = [2])
                target = torch.flip(target ,dims= [2])

        return input, target, metas, bev_maps, aux_datas

@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuplan_hr_mini_woBev(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):

        input, target, bev_maps, metas, aux_datas= self.point_cloud_dataset[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        if self.phase == 'train':
            if np.random.rand(1)<=0.5:
                input = torch.flip(input, dims = [2])
                target = torch.flip(target ,dims= [2])

        return input, target, metas