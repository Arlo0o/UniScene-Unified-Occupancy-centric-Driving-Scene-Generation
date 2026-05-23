import torch
from functools import partial
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
# from .kitti.kitti_dataset import KittiDataset
# from .nuscenes.nuscenes_dataset import NuScenesDataset
# from .waymo.waymo_dataset import WaymoDataset
# from .pandaset.pandaset_dataset import PandasetDataset
# from .lyft.lyft_dataset import LyftDataset
# from .once.once_dataset import ONCEDataset
# from .argo2.argo2_dataset import Argo2Dataset
# from .custom.custom_dataset import CustomDataset
# from .nuscenes_occ.occ_lidar import Occ2LiDARDataset
# from .nuscenes_occ.occ_lidar_nksr import Occ2LiDARDatasetNKSR
# from .nuscenes_occ.occ_lidar_gen import Occ2LiDARDatasetNKSRGen
# from .nuscenes_occ.occ_lidar_vis import Occ2LiDARDatasetVis
# from .nuscenes_occ.occ_lidar_gen_video import Occ2LiDARDatasetNKSRGenVideo
# from .nuscenes_occ.occ_lidar_nksr_customvis import Occ2LiDARDatasetNKSRCustomVis
# from .nuscenes_occ.openscene import OpenSceneDataset
# from .nuscenes_occ.waymo import WaymoOccDataset
from .nuscenes_occ.nuplan import NuPlanOccDataset
from .nuscenes_occ.nuplan_temporal import NuPlanOccDatasetTemporal
from .nuscenes_occ.nuplan_temporal_vis import NuPlanOccDatasetTemporalVis
from .nuscenes_occ.nuplan_temporal_eval import NuPlanOccDatasetTemporalEval
from .nuscenes_occ.nuplan_gen import NuPlanOccDatasetGen

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    # 'KittiDataset': KittiDataset,
    # 'NuScenesDataset': NuScenesDataset,
    # 'WaymoDataset': WaymoDataset,
    # 'PandasetDataset': PandasetDataset,
    # 'LyftDataset': LyftDataset,
    # 'ONCEDataset': ONCEDataset,
    # 'CustomDataset': CustomDataset,
    # #'Argo2Dataset': Argo2Dataset,
    # 'Occ2LiDARDataset': Occ2LiDARDataset,
    # 'Occ2LiDARDatasetNKSR': Occ2LiDARDatasetNKSR,
    # 'Occ2LiDARDatasetNKSRGen': Occ2LiDARDatasetNKSRGen,
    # 'Occ2LiDARDatasetVis': Occ2LiDARDatasetVis,
    # 'Occ2LiDARDatasetNKSRGenVideo': Occ2LiDARDatasetNKSRGenVideo,
    # 'Occ2LiDARDatasetNKSRCustomVis': Occ2LiDARDatasetNKSRCustomVis,
    # 'OpenSceneDataset': OpenSceneDataset,
    # 'WaymoOccDataset': WaymoOccDataset,
    'NuPlanOccDataset': NuPlanOccDataset,
    'NuPlanOccDatasetTemporal': NuPlanOccDatasetTemporal,
    'NuPlanOccDatasetTemporalEval': NuPlanOccDatasetTemporalEval,
    'NuPlanOccDatasetTemporalVis': NuPlanOccDatasetTemporalVis,
    'NuPlanOccDatasetGen': NuPlanOccDatasetGen
}

class CustomSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        print(f'get{idx}')
        real_idx = self.indices[idx]
        return self.dataset[real_idx]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        # print(indices)

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, subset_len=None):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    # for debug
    if subset_len is not None:
        subset_indices = list(range(subset_len))
        dataset = CustomSubsetDataset(dataset, subset_indices)

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler
