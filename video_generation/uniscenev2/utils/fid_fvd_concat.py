import torch
import json
import numpy as np
from scipy.linalg import sqrtm
from os import listdir
from os.path import isfile, join
import os
from IPython import embed
import torch
import os
import math
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from typing import Tuple
from torch.nn.functional import interpolate
from torchvision import transforms
from PIL import Image

# from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torchvision.transforms as TF
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
#############-------------------------fid
# python -m pytorch_fid  /data/proj/libohan/gitlab/drive_scene/outputs/output_0903_tos_sample/virtual/images    /data/proj/libohan/gitlab/drive_scene/outputs/output_0903_tos_sample/real/images  --dims 2048


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img
def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr
def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1,length=-1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.rglob('*.{}'.format(ext))])[:length]
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)
    
def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1, length=-1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers, length=length)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers, length=length)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1, length=-1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers, length=length)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers, length=length)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

#############-------------------------fvd

# https://github.com/universome/fvd-comparison
def load_i3d_pretrained(device=torch.device('cpu')):
    # i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    # filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
    filepath = "/gpfs/public-shared/fileset-groups/crosshair/libohan/code_0920/drive_scene/ckpts/i3d_torchscript.pt"
    print(filepath)
    if not os.path.exists(filepath):
        print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    i3d = torch.jit.load(filepath).eval().to(device)
    i3d = torch.nn.DataParallel(i3d)
    return i3d
    

def get_feats(videos, detector, device, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    feats = np.empty((0, 400))
    with torch.no_grad():
        # import pdb; pdb.set_trace()
        for i in range((len(videos)-1)//bs): # + 1):
            feats = np.vstack([feats, detector(torch.stack([preprocess_single(video) for video in videos[i*bs:(i+1)*bs]]).to(device), **detector_kwargs).detach().cpu().numpy()])
    return feats


def get_fvd_feats(videos, i3d, device, bs=10):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats(videos, i3d, device, bs)
    return embeddings


def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    if feats_fake.shape[0]>1:
        s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    else:
        fid = np.real(m)
    return float(fid)
  


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, method='styleganv'):
    print("calculate_fvd...")
    # videos [batch_size, timestamps, channel, h, w]
    assert videos1.shape == videos2.shape
    i3d = load_i3d_pretrained(device=device)
    fvd_results = []
    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]
    videos1 = trans(videos1)
    videos2 = trans(videos2)
    fvd_results = {}
    # for calculate FVD, each clip_timestamp must > 
    for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
       
        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, : clip_timestamp]
        videos_clip2 = videos2[:, :, : clip_timestamp]

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
      
        # calculate FVD when timestamps[:clip]
        fvd_results[clip_timestamp] = frechet_distance(feats1, feats2)

    result = {
        "value": fvd_results,
        "video_setting": videos1.shape,
        "video_setting_name": "batch_size, channel, time, heigth, width",
    }
    return result


class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
        self.transform = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize((64, 64)),
                          transforms.ToTensor(),
                          ])
        
    def load_video(self, filepath):
        cap = cv2.VideoCapture(filepath)
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            # 假设frame是BGR格式，将其转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame  )
            frames.append(frame)
        cap.release()
        return frames
    def __len__(self):
        return len(self.video_files)
    def __getitem__(self, idx):
        filepath = self.video_files[idx]
        frames = self.load_video(filepath)
        # 转换为torch张量
        frames_tensor = torch.stack([ frame  for frame in frames])
        return frames_tensor

      
def  video_to_tensor(video_dir, maxnum=-1, target_length=10 ):
    dataset = VideoDataset(video_dir)
 
    # 转换每一帧为张量
    tensor_frames = []
    index=0
    for frame in tqdm(dataset):
        if index >= maxnum: break 
        if len(frame) < target_length:
          frame = torch.cat(( frame,  torch.zeros( (target_length - len(frame),)+ tuple(frame.shape[1:]) ) ), dim=0)
                
        tensor_frames.append(frame)
        index+=1 
    
    # 将帧堆叠成一个四维张量
    all_videos = torch.stack((tensor_frames ), dim=0)


    # # 创建一个空的张量来存储所有视频帧
    # all_videos = None
    # for i in tqdm(range(len(dataset))) :
    #     video_tensor = dataset[i]
    #     if all_videos is None:
    #         all_videos = video_tensor.unsqueeze(0)
    #     else:
    #         all_videos = torch.cat((all_videos, video_tensor.unsqueeze(0)), dim=0)
    #     if i>maxnum: break 
    # all_videos = interpolate( all_videos.permute(0, 1,4,2,3).float().cuda(), size=(3, 64, 64), mode='trilinear')
    return all_videos




def vis_bev(input_dir, bev_map):
    import sys
    sys.path.append("/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/code/occworld-dev_12hz/")
    from  map_visualizer import visualize_map
    map_img_np=visualize_map(bev_map)
    map_img=Image.fromarray(map_img_np)
    map_img.save(os.path.join(input_dir, "bev_map.png"))
    
def concat_video(folders = ['outputs/output_sample_1022_lane/virtual/videos', 'outputs/output_sample_1022_lane/real/occ_depth', 'outputs/output_sample_1022_lane/real/occ_semantic'], 
                 output_folder = 'outputs/output_sample_1022_lane/concat_video_semantic_depth/'):
        from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
        import os        
        # 确保输出目录存在
        os.makedirs(output_folder, exist_ok=True)
        # 获取第一个文件夹中的所有视频文件名
        video_filenames = [f for f in os.listdir(folders[0]) if f.endswith(('.mp4', '.avi', '.mkv'))]
        for video_filename in video_filenames:
            # 读取每个文件夹中的同名视频
            clips = [VideoFileClip(os.path.join(folder, video_filename)) for folder in folders]
            # 获取视频的高度和宽度
            width, height = clips[0].size
            # 创建一个列表来存储复合视频剪辑的元素
            composite_clips = []
            y_offset = 0
            for clip in clips:
                # 将每个视频添加到复合视频剪辑中，设置其位置
                composite_clips.append(clip.set_position((0, y_offset)))
                # 更新下一个视频的y偏移量
                y_offset += height
            # 创建复合视频剪辑
            final_clip = CompositeVideoClip(composite_clips, size=(width, len(clips) * height))
            # 保存新的视频文件
            output_path = os.path.join(output_folder, video_filename)
            final_clip.write_videofile(output_path, codec='libx264')
            # 释放资源
            for clip in clips:
                clip.close()
            final_clip.close()
        print("视频处理完成！")


def fid_fvd(fid=1, fvd = 1, img_paths=["outputs/output_sample_1022_lane/real/images/", "outputs/output_sample_1022_lane/virtual/images/"],
            img_length=671*8*6, video_length = 671,
            video_paths=["outputs/output_sample_1022_lane/real/videos/", "outputs/output_sample_1022_lane/virtual/videos/"]):
        if fid:
            fid_score = calculate_fid_given_paths(img_paths, batch_size=128, dims=2048, device='cuda', length= img_length )
        print(f"FID Score: {fid_score}")
        if fvd:
            videos1,videos2 = video_to_tensor(video_paths[0], maxnum = video_length )  ,\
                video_to_tensor(video_paths[1], maxnum = video_length )
            print(videos1.shape,videos2.shape)
            # NUMBER_OF_VIDEOS = 8
            # VIDEO_LENGTH = 10
            # CHANNEL = 3
            # SIZE = 64
            # videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
            # videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
            device = torch.device("cuda")
            result = calculate_fvd(videos1, videos2, device, method='styleganv')
            print(json.dumps(result, indent=4))

            
if __name__ == "__main__":
    fidfvd = 0
    concatvideo = 1
    
    if fidfvd:
        fid_fvd(fid=1, fvd = 1, img_paths=["outputs/output_sample_1031_400lane_new/real/images/", "outputs/output_sample_1031_400lane_new/virtual/images/"],
            img_length=671*8*6, video_length = 671,
            video_paths=["outputs/output_sample_1031_400lane_new/real/videos/", "outputs/output_sample_1031_400lane_new/virtual/videos/"])
    if concatvideo:
        concat_video(folders = [ 'outputs/output_sample_1101_good_all/real/occ_depth/', 'outputs/output_sample_1101_good_all/real/occ_semantic/', 'outputs/output_sample_1101_good_all/virtual/videos/', ], 
                 output_folder = 'outputs/output_sample_1101_good_all/concat_video_semantic_depth/')
