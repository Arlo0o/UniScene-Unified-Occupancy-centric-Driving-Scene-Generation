
from mmengine.registry import Registry
OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')

from .dataset import nuScenesSceneDatasetLidar, nuScenesSceneDatasetLidarTraverse,nuScenesSceneDatasetLidar_ori, nuScenesSceneDatasetLidar_OpenScene
from .dataset_wrapper import tpvformer_dataset_nuscenes, custom_collate_fn_temporal
from .sampler import CustomDistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
import yaml
import pickle
import torch.multiprocessing as mp


def get_dataloader(
    train_dataset_config, 
    val_dataset_config, 
    train_wrapper_config,
    val_wrapper_config,
    train_loader, 
    val_loader, 
    rank,
    nusc=dict(
        version='v1.0-trainval',
        dataroot='data/nuscenes'),
    dist=False,
    iter_resume=False,
    train_sampler_config=dict(
        shuffle=False,
        drop_last=True),
    val_sampler_config=dict(
        shuffle=False,
        drop_last=False),
):
    train_dataset = OPENOCC_DATASET.build(
        train_dataset_config)
    val_dataset = OPENOCC_DATASET.build(
        val_dataset_config)
    
    train_wrapper = OPENOCC_DATAWRAPPER.build(
        train_wrapper_config,
        default_args={'in_dataset': train_dataset})
    val_wrapper = OPENOCC_DATAWRAPPER.build(
        val_wrapper_config,
        default_args={'in_dataset': val_dataset})
    
    train_sampler = val_sampler = None
    if dist:
        if iter_resume:
            train_sampler = CustomDistributedSampler(train_wrapper, **train_sampler_config)
        else:
            train_sampler = DistributedSampler(train_wrapper,rank=rank, **train_sampler_config)
        val_sampler = DistributedSampler(val_wrapper,rank=rank, **val_sampler_config)

    train_dataset_loader = DataLoader(
        dataset=train_wrapper,
        batch_size=train_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False if dist else train_loader["shuffle"],
        sampler=train_sampler,
        num_workers=train_loader["num_workers"],
        pin_memory=True)
    val_dataset_loader = DataLoader(
        dataset=val_wrapper,
        batch_size=val_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_loader["num_workers"],
        pin_memory=True)

    return train_dataset_loader, val_dataset_loader


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels'][val_]
    return nuScenes_label_name


def get_val_dataloader(
    train_dataset_config, 
    val_dataset_config, 
    train_wrapper_config,
    val_wrapper_config,
    train_loader, 
    val_loader, 
    rank,
    nusc=dict(
        version='v1.0-trainval',
        dataroot='data/nuscenes'),
    dist=False,
    iter_resume=False,
    train_sampler_config=dict(
        shuffle=True,
        drop_last=True),
    val_sampler_config=dict(
        shuffle=False,
        drop_last=False),
):
    val_dataset = OPENOCC_DATASET.build(
        val_dataset_config)
    

    val_wrapper = OPENOCC_DATAWRAPPER.build(
        val_wrapper_config,
        default_args={'in_dataset': val_dataset})
    
    train_sampler = val_sampler = None
    val_sampler_config['shuffle'] = val_loader["shuffle"]
    if dist:
        val_sampler = DistributedSampler(val_wrapper,rank=rank, **val_sampler_config)

    print(val_loader["shuffle"])
    val_dataset_loader = DataLoader(
        dataset=val_wrapper,
        batch_size=val_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        # shuffle=val_loader["shuffle"],
        sampler=val_sampler,
        num_workers=val_loader["num_workers"],
        pin_memory=True)

    return  val_dataset_loader


def get_train_dataloader(
    train_dataset_config, 
    train_wrapper_config,
    train_loader, 
    rank,
    nusc=dict(
        version='v1.0-trainval',
        dataroot='data/nuscenes'),
    dist=False,
    iter_resume=False,
    train_sampler_config=dict(
        shuffle=False,
        drop_last=True),
):
    train_dataset = OPENOCC_DATASET.build(
        train_dataset_config)

    
    train_wrapper = OPENOCC_DATAWRAPPER.build(
        train_wrapper_config,
        default_args={'in_dataset': train_dataset})

    
    train_sampler  = None
    if dist:
        if iter_resume:
            train_sampler = CustomDistributedSampler(train_wrapper, **train_sampler_config)
        else:
            train_sampler = DistributedSampler(train_wrapper,rank=rank, **train_sampler_config)

    train_dataset_loader = DataLoader(
        dataset=train_wrapper,
        batch_size=train_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False if dist else train_loader["shuffle"],
        sampler=train_sampler,
        num_workers=train_loader["num_workers"],
        pin_memory=True)


    return train_dataset_loader