# UniScenev2 LiDAR Generation

This folder contains the LiDAR generation branch of UniScenev2. The code is organized as a standalone OpenPCDet-style project under `lidar_generation/`; generated logs, compiled CUDA artifacts, and pretrained weights are not checked in.

## Framework

<div align=center><img width="900" src="../assets/model_lidar.png"/></div>

The LiDAR branch transfers semantic occupancy grids into sensor-specific point clouds. It uses an occupancy-conditioned sparse 3D backbone and ray-based rendering head to model LiDAR geometry, intensity, and ray interactions.

## Directory Layout

```text
lidar_generation/
├── pcdet/        # OpenPCDet-based model, dataset, CUDA ops, and utilities
├── tools/        # train / evaluation / visualization entry points
├── nuplan_sample72_r400_intenw10_pluckeremb_histemb_smlosscos_lidarcond_flim_allloc_fixrange.yaml
├── run.sh        # default UniScenev2 LiDAR inference entry
├── requirements.txt
└── setup.py
```

## Installation

Create a Python 3.9/3.10 environment and install the dependencies matching your CUDA build.

```bash
cd lidar_generation
pip install -r requirements.txt
pip install -e . -v
```

Install the DDA CUDA extension used by the NuPlan occupancy-to-LiDAR dataset code.

```bash
cd pcdet/datasets/nuscenes_occ/utils/dda
pip install . -v
```

For OpenPCDet-specific environment issues, refer to the official [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) installation guide.

## Pretrained Model

Download the released UniScenev2 LiDAR checkpoint from the NuPlan-Occ Hugging Face dataset.

```bash
cd lidar_generation
pip install -U huggingface_hub
huggingface-cli download Arlolo0/Nuplan-Occupancy \
  --repo-type dataset \
  --include "checkpoint/lidar_generation/*" \
  --local-dir .
```

The checkpoint should be placed as:

```text
lidar_generation/checkpoint/lidar_generation/checkpoint.pth
```

## Data Preparation

Prepare NuPlan sensor data, NuPlan-Occ pickle files, and quantized occupancy grids.

```text
lidar_generation/
├── data/
│   ├── nuplan_all/
│   │   └── sensor_blobs/
│   │       └── mini/
│   ├── nuplan_pkls/
│   │   └── mini/
│   │       ├── nuplan_mini_10hz_train.pkl
│   │       └── nuplan_mini_10hz_val.pkl
│   └── occ_quan/
│       └── nuplan_quantized_400_400_32/
```

The default dataset config is `tools/cfgs/dataset_configs/nuplan_dataset_r400.yaml`. Adjust `DATA_PATH`, `lidar_path`, `pkl_path`, `val_pkl_path`, and `occ_path` if your data layout differs.

## Inference

Run the default UniScenev2 LiDAR generation config.

```bash
cd lidar_generation
bash run.sh
```

Equivalent explicit command:

```bash
python tools/test.py \
  --cfg_file nuplan_sample72_r400_intenw10_pluckeremb_histemb_smlosscos_lidarcond_flim_allloc_fixrange.yaml \
  --ckpt checkpoint/lidar_generation/checkpoint.pth \
  --work_dir outputs/lidar_generation \
  --save_to_file
```

Override data paths without editing the YAML by appending OpenPCDet config overrides after `--set`:

```bash
python tools/test.py \
  --cfg_file nuplan_sample72_r400_intenw10_pluckeremb_histemb_smlosscos_lidarcond_flim_allloc_fixrange.yaml \
  --ckpt checkpoint/lidar_generation/checkpoint.pth \
  --work_dir outputs/lidar_generation \
  --save_to_file \
  --set DATA_CONFIG.DATA_PATH /path/to/nuplan_all \
        DATA_CONFIG.val_pkl_path /path/to/nuplan_mini_10hz_val.pkl \
        DATA_CONFIG.occ_path /path/to/nuplan_quantized_400_400_32
```

## Training

Use `tools/train.py` with a selected config:

```bash
cd lidar_generation
torchrun --nproc_per_node 8 tools/train.py \
  --cfg_file nuplan_sample72_r400_intenw10_pluckeremb_histemb_smlosscos_lidarcond_flim_allloc_fixrange.yaml \
  --work_dir outputs/lidar_train
```

## Acknowledgements

This implementation builds on excellent open-source projects including [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), NeuS, and NeuRAD.
