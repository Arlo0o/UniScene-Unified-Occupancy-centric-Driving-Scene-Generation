## UniScene: Unified Occupancy-centric Driving Scene Generation [CVPR 2025]



 [![arXiv paper](https://img.shields.io/badge/arXiv%20%2B%20supp-2412.05435-purple)](https://arxiv.org/abs/2412.05435) 
[![Code page](https://img.shields.io/badge/Project%20Page-UniScene-red)](https://arlo0o.github.io/uniscene/)
[![Code page](https://img.shields.io/badge/PDF%20File-UniScene-green)](./assets/UniScene-arxiv.pdf)
[![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/Arlolo0/UniScene/tree/main) 

### 🎯 Demo:
<div align=center><img width="960" height="230" src="./assets/teaser_fig1.png"/></div>

**(a) Overview of UniScene.** Given BEV layouts, UniScene facilitates versatile data generation, including semantic occupancy, multi-view video, and LiDAR point clouds, through an occupancy-centric hierarchical modeling approach. **(b) Performance comparison on different generation tasks.** UniScene delivers substantial improvements over SOTA methods in video, LiDAR, and occupancy generation.



<!-- <div align=center><img width="960" height="470" src="./assets/teaser_fig1_b.png"/></div>
 Versatile generation ability of UniScene.  -->

<br>

<div align=center><img width="640" height="380" src="./assets/demo.gif"/></div>



### 📋 Abstract:
<details>
<summary><b>TL; DR</b>  The first unified framework for generating three key data forms — semantic occupancy, video, and LiDAR — in driving scenes. </summary>

Generating high-fidelity, controllable, and annotated training data is critical for autonomous driving. Existing methods typically generate a single data form directly from a coarse scene layout, which not only fails to output rich data forms required for diverse downstream tasks but also struggles to model the direct layout-to-data distribution. In this paper, we introduce UniScene, the first unified framework for generating three key data forms — semantic occupancy, video, and LiDAR — in driving scenes. UniScene employs a progressive generation process that decomposes the complex task of scene generation into two hierarchical steps: (a) first generating semantic occupancy from a customized scene layout as a meta scene representation rich in both semantic and geometric information, and then (b) conditioned on occupancy, generating video and LiDAR data, respectively, with two novel transfer strategies of Gaussian-based Joint Rendering and Prior-guided Sparse Modeling. This occupancy-centric approach reduces the generation burden, especially for intricate scenes, while providing detailed intermediate representations for the subsequent generation stages. Extensive experiments demonstrate that UniScene outperforms previous SOTAs in the occupancy, video, and LiDAR generation, which also indeed benefits downstream driving tasks.
</details>

### 📚 Framework:
<div align=center><img width="940" height="260" src="./assets/overall.png"/></div>

 

### 💥 News
- [2025/03]: Code and pre-trained weights are released.
- [2025/02]: Paper is accepted on **CVPR 2025**.
- [2024/12]: Paper is on the [arxiv](https://arxiv.org/abs/2412.05435).
- [2024/12]: Demo is released on the [Project Page](https://arlo0o.github.io/uniscene/).



### 🕹️ Getting Started


#### 1. Environment Setup
 
**a.**  We recommend to create environment using the following script with "poetry", which have been tested on NVIDIA A100/A800 with CUDA 12.1 and Python 3.9:

 
[Step-by-step Installation.](./install.md)
 


**b.** Pretrained Models. The huggingface-cli tool is available with: 
```bash
pip install -U huggingface_hub 
huggingface-cli download --resume-download Arlolo0/UniScene_path  --local-dir $local_path
```

| Model      | OneDrive&nbsp;<img src="https://img.icons8.com/?size=32&id=13638&format=png&color=000000" width="24" alt="OneDrive" style="vertical-align: middle;" />  |   [![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/Arlolo0/UniScene/tree/main)                                                                   |
|:-----------|:---------|:-----------------------------------------------------------------------------------|
| Occupancy Model    |[Occuancy-OneDrive](https://nbeitech-my.sharepoint.com/:f:/g/personal/bli_eitech_edu_cn/EpYIjg5_l2VFoYJd2vZcl9wBFeVQV1XI_NPQQhXOB-wUqQ?e=I3vmYQ)| [Occ-VAE](https://huggingface.co/Arlolo0/UniScene/resolve/main/Occupancy_Generation_ckpt_AE_eval_epoch_196.pth?download=true) [Occ-DiT](https://huggingface.co/Arlolo0/UniScene/resolve/main/Occupancy_Generation_ckpt_DiT_0600000.pt?download=true) |
| LiDAR Model     |[LiDAR-OneDrive](https://nbeitech-my.sharepoint.com/:u:/g/personal/bli_eitech_edu_cn/EcMXzN216PpHtvnfWubStf0BrqsfuNG4WSy8fmH08Qt_8Q)| [LiDAR-HF](https://huggingface.co/Arlolo0/UniScene/resolve/main/occ2lidar.pth?download=true)  |
| Video Model     |[Video-OneDrive](https://nbeitech-my.sharepoint.com/:u:/g/personal/bli_eitech_edu_cn/EUiGlxoQ3ENDksEXKSmLP_IBkgIBSXZnYsUDtEeIQQGfxg?e=ovQTGD)| [Video-HF](https://huggingface.co/Arlolo0/UniScene/resolve/main/video_pretrained.safetensors?download=true)  |

 



#### 2. Data Preparation
**a.** Download all splits of **Trainval** in **Full dataset (v1.0)** to your device following [official instructions](https://www.nuscenes.org/download) and put them to local folder "./data/nuscenes". 
```
$./data/nuscenes
├── samples
├── sweeps
├── ...
└── v1.0-trainval
```

**b.** Downloaded interpolated 12Hz annotation and mmdet3d meta data on ["nuscenes_interp_12Hz_infos_*.pkl"](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155157018_link_cuhk_edu_hk/EXunN1j0OmNLtaPoh2VrkgQBGpyXiMlltuCX5GBuYc00YQ?e=bVI9AC) and put them to "data/nuscenes_mmdet3d-12Hz/".

**c.** (Optional) Prepare 12HZ 3D Occupancy Labels with LiDAR and 3D bbox labels.
Note that we use [NKSR reconstruction](https://github.com/nv-tlabs/NKSR) to produce GT occupancy with the pc-range of [-50, -50, -5, 50, 50, 3], which is aligned with [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy).
The pc-range of the generated occupancy from our [occupancy model](occupancy_gen/README.md) is set to [-40, -40, -1, 40, 40, 5.4] for [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D) baseline comparisons.




To generate ground-truth semantic occupancy with resolution of 200×200×16:

```bash
cd ./data_process && python generate_occ.py --config_path  config-200.yaml --save_path $GT_nksr_occupancy_200/
```



To generate ground-truth semantic occupancy with resolution of 800×800×64:

```bash
cd ./data_process && python generate_occ.py --config_path  config-800.yaml --save_path $GT_nksr_occupancy_800/
```

Note that this will take up about 3.1 TB of hard disk space.




**d.** (Optional) Prepare 12HZ BEV layouts.
   
```bash
cd ./occupancy_gen/12hz_processing && python save_bevlayout_12hz.py  \
    --py-config="./config/save_step2_me.py" \
    --work-dir="./ckpt/VAE/" 
```

Note that the road lines from the BEV layouts are projected onto the semantic occupancy, integrating the corresponding semantic information. For more details, please refer to our UniScene [paper](https://arxiv.org/abs/2412.05435).

#### 3. Multi-modal Generation
- Overall running instruction:
    
    You can link the "./data" folder to the following subfolders for convenient access: 
    
    ```bash
    ln -s ./data   $sub_folder  
    ```

    **a.** Our framework starts with generating semantic occupancy from given BEV layouts as:

    ```bash
    cd ./occupancy_gen
    python save_bev_layout.py \
        --py-config="./config/save_step2_me.py" \
        --work-dir="./ckpt/VAE/" 
    bash ./run_eval_dit.sh
    ```


    **b.** Generating LiDAR point clouds from semantic occupancy as:

    ```bash
    cd ./lidar_gen
    python tools/test.py --save_to_file  $output_lidar_path   
    ```


    **c.** Generating multi-view video from semantic occupancy as:

    ```bash
    cd ./video_gen
    python  ./gs_render/render_eval_condition_gt.py  --occ_path  $input_occ_path  --layout_path $input_bev_path --render_path $output_render_path  --vis
    python  inference_video.py  --occ_data_root $output_render_path   --save  $output_video_path  
    ```


- More details on Occupancy, LIDAR, and Video generation. We leverage [Mayavi](https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-pip) for 3D occupancy and LiDAR point clouds visualization:

	-  [Occuancy Generation.](occupancy_gen/README.md) Please refer to [Occupancy_vis](occupancy_gen\visualize_nuscenes_occupancy.py) for visualization.
	
	
	-  [LiDAR Generation.](lidar_gen/README.md) Please refer to [LiDAR_vis](lidar_gen/visualize_nuscenes_lidar.py) for visualization.
	
	
	-  [Video Generation.](video_gen/README.md) The generated video will be saved at "./video_gen/outputs".

### ❤️Acknowledgements
Our implementation is based on the excellent open source projects: [Occworld](https://github.com/wzzheng/OccWorld), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [SVD](https://github.com/Stability-AI/generative-models).



### 📜 License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.





### ⭐ Citation
If you find our paper and code useful for your research, please consider citing us and giving a star to our repository:

```bibtex

@article{li2024uniscene,
  title={UniScene: Unified Occupancy-centric Driving Scene Generation},
  author={Li, Bohan and Guo, Jiazhe and Liu, Hongsi and Zou, Yingshuang and Ding, Yikang and Chen, Xiwu and Zhu, Hu and Tan, Feiyang and Zhang, Chi and Wang, Tiancai and others},
  journal={arXiv preprint arXiv:2412.05435},
  year={2024}
}
```

