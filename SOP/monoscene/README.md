# MonoScene Baseline for NuPlan-Occ Dataset

This repository provides a **reproduced baseline** of **MonoScene** for the **NuPlan-Occ** dataset.
<!-- , as part of the **WACV 2026 Challenge on 3D Semantic Occupancy Prediction for Autonomous Driving**. -->

The baseline model is trained and evaluated on the **NuPlan-Occ** dataset—the largest semantic occupancy benchmark to date, featuring **3.6 million frames** with high-resolution voxel annotations.

## 📋 **Results on NuPlan-Occ Validation Set**

| Metric | Value |
|--------|-------|
| Precision | 48.9942 |
| Recall | 42.5207 |
| IoU | 29.4737 |
| **mIoU** | **9.3487** |

**Per-class IoU**:  
`empty`: 96.0546, `background`: 28.9968, `vehicle`: 17.6794, `bicycle`: 0.2984, `pedestrian`: 6.4943, `traffic_cone`: 1.8300, `barrier`: 2.7998, `czone_sign`: 2.9211, `generic_object`: 13.7700

---

## ⚙️ **Quick Start**

### 1. Environment Setup

Create and activate the original monoscene environment:

1. Create conda environment:

```
$ conda create -y -n monoscene python=3.7
$ conda activate monoscene
```
2. This code was implemented with python 3.7, pytorch 1.7.1: 

```
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install the additional dependencies:

```
$ cd MonoScene/
$ pip install -r requirements.txt
```

4. Install tbb:

```
$ conda install -c bioconda tbb=2020.2
```

5. Downgrade torchmetrics to 0.6.0
```
$ pip install torchmetrics==0.6.0
```

6. Install MonoScene:

```
$ pip install -e ./
```


### 2. Prepare NuPlan-Occ Dataset

Follow the instructions in the [UniScene repository](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2?tab=readme-ov-file#2-prepare-data) to prepare the NuPlan-Occ dataset.

Then, link the dataset to the appropriate directories:

```bash
mkdir data
ln -s Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ data/occ_quan
ln -s Nuplan-Occupancy/pickle data/nuplan_pkls
mkdir -p ./data/nuplan_all/sensor_blobs/mini/
```

Link the camera data directories:

```bash
src_dirs=(
  "nuplan/miniset/mini_camera/nuplan-v1.1_mini_camera_0/"
  "nuplan/miniset/mini_camera/nuplan-v1.1_mini_camera_1/"
  "nuplan/miniset/mini_camera/nuplan-v1.1_mini_camera_2/"
  "nuplan/miniset/mini_camera/nuplan-v1.1_mini_camera_3/"
  "nuplan/miniset/mini_camera/nuplan-v1.1_mini_camera_4/"
  "nuplan/miniset/mini_camera/nuplan-v1.1_mini_camera_5/"
  "nuplan/miniset/mini_camera/nuplan-v1.1_mini_camera_6/"
  "nuplan/miniset/mini_camera/nuplan-v1.1_mini_camera_7/"
  "nuplan/miniset/mini_camera/nuplan-v1.1_mini_camera_8/"
)

for src_dir in "${src_dirs[@]}"; do
  find "$src_dir" -mindepth 1 -maxdepth 1 -type d | while read -r subdir; do
    dir_name=$(basename "$subdir")
    ln -s "$subdir" "./data/nuplan_all/sensor_blobs/mini/$dir_name" 2>/dev/null
  done
done

echo "Link complete: ./data/nuplan_all/sensor_blobs/mini"
```

### 3. Evaluation

1. **Download the pre-trained checkpoint**:
   - [Google Drive link](https://drive.google.com/file/d/14OPVxfvDIpZVozbOobxTsc8IEDF8UaJA/view?usp=sharing)
   - Place the downloaded checkpoint in an appropriate directory (e.g., `trained_models/`)

2. **Cache the EfficientNet model**:
   ```bash
   cp -r gen-efficientnet-pytorch /root/.cache/torch/hub/
   ```

3. **Run evaluation**:
   - For **1 GPU** (default):
     ```bash
     python monoscene/scripts/eval_monoscene.py
     ```
     Uses configuration: `monoscene/config/monoscene_nuplan_eval_1gpu.yaml`
   
   - For **4 GPUs**:
     ```bash
     python monoscene/scripts/eval_monoscene.py config=monoscene/config/monoscene_nuplan_eval.yaml
     ```


---

## 🔗 **Related Resources**
- **Main Repository**: [UniScenev2: Unified Occupancy-centric Driving Scene Generation](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2)
<!-- - **Challenge Website**: [WACV 2026 Challenge on 3D Semantic Occupancy Prediction](https://arlo0o.github.io/uniscenev2/challenge) *(to be created)* -->
- **Dataset**: [NuPlan-Occ on HuggingFace](https://huggingface.co/datasets/Arlolo0/Nuplan-Occupancy)



---

## 🛠️ **Troubleshooting**

- **Mayavi installation issues**: Refer to the [original MonoScene installation guide](https://anhquancao.github.io/blog/2022/how-to-install-mayavi-with-python-3-on-ubuntu-2004-using-pip-or-anaconda/)
- **Dataset linking problems**: Ensure all paths are correctly set and the dataset is properly downloaded
- **GPU memory issues**: Reduce batch size in the configuration files if needed

---

## 📄 **License**

This project is released under the [Apache 2.0 License](LICENSE).



