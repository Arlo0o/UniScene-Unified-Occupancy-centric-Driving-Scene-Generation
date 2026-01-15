from torch.utils.data.dataloader import DataLoader
from monoscene.data.nuplan.nuplan_dataset import NuplanDataset
import pytorch_lightning as pl
from monoscene.data.nuplan.collate import collate_fn
from monoscene.data.utils.torch_util import worker_init_fn


class NuplanDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        pkls_path,
        occ_path,
        target_width=1920,
        target_height=1080,
        project_scale=2,
        frustum_size=4,
        batch_size=4,
        num_workers=6,
    ):
        super().__init__()
        self.root = root
        self.pkls_path = pkls_path
        self.occ_path = occ_path
        self.project_scale = project_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        self.target_width = target_width
        self.target_height = target_height

    def setup(self, stage=None):
        self.train_ds = NuplanDataset(
            split="train",
            root=self.root,
            pkls_path=self.pkls_path,
            occ_path=self.occ_path,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
            target_width=self.target_width,
            target_height=self.target_height
        )

        self.val_ds = NuplanDataset(
            split="val",
            root=self.root,
            pkls_path=self.pkls_path,
            occ_path=self.occ_path,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
            target_width=self.target_width,
            target_height=self.target_height
        )

        self.test_ds = NuplanDataset(
            split="val",
            root=self.root,
            pkls_path=self.pkls_path,
            occ_path=self.occ_path,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
            target_width=self.target_width,
            target_height=self.target_height
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
