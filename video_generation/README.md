# UniScenev2 Video Generation

This folder contains the video generation code for UniScenev2. The original `pwm` package has been renamed to `uniscenev2`; scripts and registry locations have been updated accordingly.

## Framework

<div align=center><img width="900" src="../assets/model_video.png"/></div>

The video branch generates surround-view driving videos from occupancy-aware conditions. Semantic occupancy is first rendered into sparse point map/depth/semantic condition maps, then a DiT-based video diffusion model combines these maps with camera metadata and text embeddings to synthesize temporally coherent multi-view videos.

## Directory Layout

```text
video_generation/
├── config/                 # NuPlan video generation configs
├── render_release/         # occupancy-to-condition rendering scripts
├── tools/                  # train / inference / VAE utilities
├── uniscenev2/             # video generation model, dataset, scheduler, utils
├── run.sh                  # default inference entry
└── dis_train_nuplan_mul.sh # helper for multi-shard NuPlan training symlinks
```

## Installation

Create a Python 3.9 environment, then install the video dependencies.

```bash
cd video_generation
pip install -r requirements.txt
```

For the Gaussian rendering extension used to prepare semantic/depth condition maps:

```bash
cd render_release/diff-gaussian-rasterization
pip install -e . -v
```

`flash-attn`, `mmcv-full`, and CUDA/PyTorch wheels are environment-sensitive. Install the versions matching your CUDA and PyTorch build before running training or inference.

## Pretrained Models

Download the UniScenev2 video checkpoint from the NuPlan-Occ Hugging Face dataset.

```bash
cd video_generation
pip install -U huggingface_hub
huggingface-cli download Arlolo0/Nuplan-Occupancy \
  --repo-type dataset \
  --include "checkpoint/video_generation/*" \
  --local-dir .
```

The checkpoint should be placed as:

```text
video_generation/checkpoint/video_generation/
├── config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001.bin
└── pytorch_model-00002.bin
```

The VAE and text encoder are not redistributed here. Download them from the official model repositories and keep the paths expected by the configs:

```bash
cd video_generation
huggingface-cli download THUDM/CogVideoX-2b \
  --local-dir ckpts/cogvideox-2b
huggingface-cli download google/t5-v1_1-xxl \
  --local-dir ckpts/t5-v1_1-xxl
```

`THUDM/CogVideoX-2b` currently redirects to `zai-org/CogVideoX-2b` on Hugging Face; either model id works with the Hugging Face CLI.

## Data Preparation

Prepare NuPlan camera data, NuPlan-Occ pickle files, and rendered condition maps.

```text
video_generation/
├── dataset1/
│   └── nuplan/
│       └── sensor_blobs_train/
├── dataset1/
│   └── nuplan-occ-render-mini/
│       └── mini/
└── pickle/
    └── nuplan_mini_10hz_pkl/
        ├── nuplan_mini_train.pkl
        └── nuplan_mini_val.pkl
```

If you downloaded the pickles from the main UniScenev2 dataset layout, you can symlink them.

```bash
mkdir -p video_generation/pickle/nuplan_mini_10hz_pkl
ln -s ../../data/nuplan_pkls/mini/nuplan_mini_10hz_train.pkl \
  video_generation/pickle/nuplan_mini_10hz_pkl/nuplan_mini_train.pkl
ln -s ../../data/nuplan_pkls/mini/nuplan_mini_10hz_val.pkl \
  video_generation/pickle/nuplan_mini_10hz_pkl/nuplan_mini_val.pkl
```

To build camera-data symlinks from multiple NuPlan shards.

```bash
cd video_generation
NUPLAN_CAMERA_ROOTS="/path/to/camera_shard_0:/path/to/camera_shard_1" \
OCC_RENDER_ROOT="/path/to/nuplan-occ-render-mini" \
bash dis_train_nuplan_mul.sh
```

## Render Condition Maps

The video model uses sparse point map/depth/semantic condition maps. Render them from occupancy and BEV layout data.

```bash
cd video_generation
bash render_release/run_render_nuplan_val_nomap.sh 0 500
```

Override paths as needed:

```bash
OCC_PATH=/path/to/dense_voxels_with_semantic \
LAYOUT_PATH=/path/to/nuplan_bev \
DATASET_PATH=/path/to/nuplan/sensor_blobs/trainval \
RENDER_PATH=/path/to/output_render \
bash render_release/run_render_nuplan_val_nomap.sh 0 500
```

## Inference

After preparing checkpoints, VAE/T5, pickles, camera data, and rendered condition maps.

```bash
cd video_generation
bash run.sh
```

Equivalent explicit command:

```bash
torchrun --master_port 29507 --nproc_per_node 1 \
  tools/inference.py config/stage_3_video_pretrain_dit3d_nuplan_control_all_10hz_single_sample.py \
  --sample-every 1 \
  --load checkpoint/video_generation \
  --output outputs/video_generation
```

## Training

Stage configs are kept under `config/`. The main NuPlan control configs are:

- `config/stage_2_video_pretrain_dit3d_nuplan_control_all_10hz.py`
- `config/stage_2_video_pretrain_dit3d_nuplan_control_all_10hz_single.py`
- `config/stage_3_video_pretrain_dit3d_nuplan_control_all_10hz_single.py`
- `config/stage_3_video_pretrain_dit3d_nuplan_control_all_10hz_single_sample.py`

Launch with:

```bash
cd video_generation
torchrun --master_port 29512 --nproc_per_node 8 \
  tools/train.py config/stage_3_video_pretrain_dit3d_nuplan_control_all_10hz_single.py
```

Adjust `img_path`, `seg_path`, `depth_path`, pickle paths, and `load` in the selected config if your data layout differs.

## Acknowledgements

This implementation builds on excellent open-source projects including CogVideoX, Colossal-AI, Diffusers, MMDetection3D, and Gaussian Splatting rasterization utilities.
