# 安装nuplan-devkit
需要 python3.9 以上. 
```
# 安装mmdetedtion3d环境
conda create -n occworld python=3.9 -y
conda activate occworld
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# change some package
# pip install numpy==1.23.0 
pip install einops scipy==1.10.1 scikit-video nvitop cmake lit timm

pip install -U openmim
mim install mmengine mmcv==2.1.0 mmdet==3.3.0
# mim install mmcv==2.1.0 mmdet==3.3.0
# mim install mmdet==3.3.0


git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
cd mmdetection3d
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in edtiable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```