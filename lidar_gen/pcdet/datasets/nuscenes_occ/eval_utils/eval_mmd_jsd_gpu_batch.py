import os

import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from jsd import JensenShannonDivergence
from mmd_gpu import MaximumMeanDiscrepancy
from open3d_utils import render_open3d, save_open3d_render
from unvoxelize import * 

from dataset_utils import voxelize
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

#SPATIAL_RANGE = [-50, 50, -50, 50, -3.23, 3.77]
#SPATIAL_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
SPATIAL_RANGE = [-51.2, 51.2, -51.2, 51.2, -5.0, 3.0]
VOXEL_SIZE = [0.15625, 0.15625, 0.2]
JSD_SHAPE = [1, 100, 100]

class NPYDataset(Dataset):
    def __init__(self, filenames1, filenames2, rotations=None, flip_vert=False) -> None:
        super().__init__()
        self.filenames1 = filenames1
        self.filenames2 = filenames2
        self.rotations = rotations
        self.flip_vert = flip_vert
        assert len(self.filenames1) == len(self.filenames2)

    def __len__(self):
        return len(self.filenames1)

    def __getitem__(self, i):
        pts1 = np.load(self.filenames1[i])
        if pts1.shape[1] > 3:
            pts1 = pts1[:, :3]

        pts2 = np.load(self.filenames2[i])
        if pts2.shape[1] > 3:
            pts2 = pts2[:, :3]
        return pts1, pts2
        #return load_npy_and_voxelize(self.filenames1[i]), load_npy_and_voxelize(self.filenames2[i], rotations=self.rotations, flip_vert=self.flip_vert)

    @staticmethod
    def collect_fn(data):
        batch_pts1, batch_pts2 = zip(*data)
        coors = []
        for i, coor in enumerate(batch_pts1):
            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
            coors.append(coor_pad)
        batch_pts1 = np.concatenate(coors, axis=0)

        coors = []
        for i, coor in enumerate(batch_pts2):
            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
            coors.append(coor_pad)
        batch_pts2 = np.concatenate(coors, axis=0)
        return torch.from_numpy(batch_pts1), torch.from_numpy(batch_pts2)

def load_npy(filename):
    pts = np.load(filename)
    if pts.shape[1] > 3:
        pts = pts[:, :3]
    return pts

def load_npy_and_voxelize(filename, rotations=None, flip_vert=False):
    pts = np.load(filename)
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    voxelized = voxelize(pts, SPATIAL_RANGE, VOXEL_SIZE)
    voxelized = np.transpose(voxelized, (2, 0, 1))

    if(rotations is not None):
        voxelized = np.rot90(voxelized, k=rotations, axes=(1,2)).copy()

    if(flip_vert):
        voxelized = np.flip(voxelized, axis=2).copy()

    return voxelized


def _voxelize(pts, rotations=None, flip_vert=False):

    voxelized = voxelize(pts, SPATIAL_RANGE, VOXEL_SIZE)
    voxelized = np.transpose(voxelized, (2, 0, 1))

    if(rotations is not None):
        voxelized = np.rot90(voxelized, k=rotations, axes=(1,2)).copy()

    if(flip_vert):
        voxelized = np.flip(voxelized, axis=2).copy()

    return voxelized


def _voxelize_gpu(batch_pts, rotations=None, flip_vert=False):
    spatial_range = SPATIAL_RANGE
    voxel_size = VOXEL_SIZE
    dtype = torch.float32
    spatial_range = torch.tensor(spatial_range, dtype=batch_pts.dtype).cuda()
    voxel_size = torch.tensor(voxel_size, dtype=batch_pts.dtype).cuda()

    # Get coordinate min and max
    coords_min, coords_max = spatial_range[[0, 2, 4]], spatial_range[[1, 3, 5]]

    # Quantize coordinates
    batch_size = int(batch_pts[-1, 0]) + 1
    results = []
    for bs in range(batch_size):
        batch_mask = batch_pts[:, 0].int()==bs
        pts = batch_pts[batch_mask][:, 1:]
        coords = ((pts - coords_min) / voxel_size).to(torch.int32)
        coords = torch.unique(coords, dim=0)

        # Create volume
        volume_size = torch.ceil((coords_max - coords_min) / voxel_size).to(torch.int32)
        volume = torch.zeros(volume_size.tolist(), dtype=dtype, device=coords.device)

        # Remove points outside the volume
        mask = torch.all((coords[:] >= 0) & (coords[:] < volume_size[:]), dim=1)
        coords = coords[mask].long()

        # Fill volume
        volume[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

        voxelized = volume
        # voxelized = voxelize(pts, SPATIAL_RANGE, VOXEL_SIZE)
        voxelized = voxelized.permute((2, 0, 1))
        # voxelized = np.transpose(voxelized, (2, 0, 1))

        if(rotations is not None):
            voxelized = torch.rot90(voxelized, k=rotations, dims=(1,2)).clone()

        if(flip_vert):
            voxelized = torch.flip(voxelized, dims=2).clone()

        results.append(voxelized)
    return results


def sanity_visualize(in_filename, out_filename, rotations=None):
    voxelized = load_npy_and_voxelize(in_filename, rotations)

    voxelized_bev = voxelized.sum(0)

    plt.imshow(voxelized_bev)
    plt.savefig(out_filename)

def main() -> None:

    parser = argparse.ArgumentParser(
        'Eval Set'
    )

    parser.add_argument('folder1', type=str) #Intended to be Ground Truth
    parser.add_argument('folder2', type=str) #Intended to be samples
    parser.add_argument('--type', type=str, default="jsd") #jsd, mmd, viz
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--folder2_rotations', default=0, type=int) #Number of rot90's to apply 
    parser.add_argument('--folder2_flip_vert', default=False, action='store_true')
    parser.add_argument('--viz_folder', type=str, default="viz")
    parser.add_argument('--filter_pts_range', action='store_true', default=False)
    parser.add_argument('--tag', type=str, default="")

    args=parser.parse_args()

    if args.filter_pts_range:
        print('Limit point cloud range to radius 3-50')

    filenames1 = glob(args.folder1 + '/*')
    filenames2 = glob(args.folder2 + '/*')

    # filenames1 = filenames1[:123]
    # filenames2 = filenames2[:123]

    sanity_visualize(filenames1[0], "f1_0.png")
    sanity_visualize(filenames1[1], "f1_1.png")
    sanity_visualize(filenames1[2], "f1_2.png")

    sanity_visualize(filenames2[0], "f2_0.png", rotations=args.folder2_rotations)
    sanity_visualize(filenames2[1], "f2_1.png", rotations=args.folder2_rotations)
    sanity_visualize(filenames2[2], "f2_2.png", rotations=args.folder2_rotations)

    if(args.type == "viz"):

        os.system(f"mkdir {args.viz_folder}")
        os.system(f"mkdir {args.viz_folder}/set1")
        os.system(f"mkdir {args.viz_folder}/set2")

        for i in tqdm(range(0, len(filenames2))):
            unvoxelized1 = load_npy(filenames1[i])
            unvoxelized2 = load_npy(filenames2[i])

            #voxelized1 = load_npy_and_voxelize(filenames1[i])
            #voxelized2 = load_npy_and_voxelize(filenames2[i], rotations=args.folder2_rotations, flip_vert=args.folder2_flip_vert)

            #voxelized1 = np.rot90(voxelized1, k=3, axes=(1,2)).copy()
            #voxelized2 = np.rot90(voxelized2, k=3, axes=(1,2)).copy()

            #unvoxelized1 = unvoxelize(torch.from_numpy(voxelized1), SPATIAL_RANGE, VOXEL_SIZE).detach().cpu().numpy()
            #unvoxelized2 = unvoxelize(torch.from_numpy(voxelized2), SPATIAL_RANGE, VOXEL_SIZE).detach().cpu().numpy()
            
            bev_img1, pts_img1, side_img1 = render_open3d(unvoxelized1, SPATIAL_RANGE, ultralidar=True)
            bev_img2, pts_img2, side_img2 = render_open3d(unvoxelized2, SPATIAL_RANGE, ultralidar=True)



            save_open3d_render(f"{args.viz_folder}/set1/{i}_side.png", side_img1, quality=9)
            save_open3d_render(f"{args.viz_folder}/set1/{i}_pts.png", pts_img1, quality=9) 
            save_open3d_render(f"{args.viz_folder}/set2/{i}_side.png", side_img2, quality=9)
            save_open3d_render(f"{args.viz_folder}/set2/{i}_pts.png", pts_img2, quality=9) 



    else:

        # if(args.type == "jsd"):
        #     metric = JensenShannonDivergence(JSD_SHAPE)
        # else:
        #     metric = MaximumMeanDiscrepancy()

        metric_jsd = JensenShannonDivergence(JSD_SHAPE).cuda()
        metric_mmd = MaximumMeanDiscrepancy().cuda()

        # for i in tqdm(range(0, len(filenames2))):
        #     voxelized1 = load_npy_and_voxelize(filenames1[i])
        #     voxelized2 = load_npy_and_voxelize(filenames2[i], rotations=args.folder2_rotations, flip_vert=args.folder2_flip_vert)

        #     data_map = {
        #         "lidar": torch.from_numpy(voxelized1).unsqueeze(0),
        #         "sample": torch.from_numpy(voxelized2).unsqueeze(0)
        #     }

        #     metric.update(data_map)

        dataloader = DataLoader(NPYDataset(filenames1, filenames2, args.folder2_rotations, args.folder2_flip_vert), batch_size=8, shuffle=False, num_workers=8, collate_fn=NPYDataset.collect_fn)

        for pts1, pts2 in tqdm(dataloader):
            pts1 = pts1.cuda()
            pts2 = pts2.cuda()

            if args.filter_pts_range:
                # bxyz
                dis1 = torch.linalg.norm(pts1[:, 1:4], dim=-1)
                dis_mask1 = (dis1 > 3.0) & (dis1 < 50.0)
                pts1 = pts1[dis_mask1]
                
                dis2 = torch.linalg.norm(pts2[:, 1:4], dim=-1)
                dis_mask2 = (dis2 > 3.0) & (dis2 < 50.0)
                pts2 = pts2[dis_mask2]

            batch_voxelized1 = _voxelize_gpu(pts1)
            batch_voxelized2 = _voxelize_gpu(pts2, rotations=args.folder2_rotations, flip_vert=args.folder2_flip_vert)

            for voxelized1, voxelized2 in zip(batch_voxelized1, batch_voxelized2):
                data_map = {
                    "lidar": voxelized1.unsqueeze(0),
                    "sample": voxelized2.unsqueeze(0)
                }

                metric_jsd.update(data_map)
                metric_mmd.update(data_map)

        jsd_save_path = os.path.join(os.path.dirname(args.folder2), f'jsd_middle{args.tag}.pkl')
        mmd_save_path = os.path.join(os.path.dirname(args.folder2), f'mmd_middle{args.tag}.pkl')
        print(f'Comparing folder1 (GT):  {args.folder1}')
        print(f'     to   folder2 (Gen): {args.folder2}')

        metric_score_jsd = metric_jsd.compute(middle_save_path=jsd_save_path)
        print(f'jsd score is: {str(metric_score_jsd)}')
        
        metric_score_mmd = metric_mmd.compute(middle_save_path=mmd_save_path)
        scientific_notation = "{:e}".format(metric_score_mmd)
        print(f'mmd score is: {scientific_notation}')

        if args.log_file is not None:
            with open(args.log_file, 'w') as f:
                print(f'Comparing folder1 (GT):  {args.folder1}\n', file=f)
                print(f'     to   folder2 (Gen): {args.folder2}\n', file=f)

                scientific_notation = "{:e}".format(metric_score_mmd)
                print(f'mmd score is: {scientific_notation}\n', file=f)
                print(f'jsd score is: {str(metric_score_jsd)}\n', file=f)


if __name__ == "__main__":
    main()
