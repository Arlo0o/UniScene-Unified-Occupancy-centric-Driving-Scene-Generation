import os

import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from chamferdist import ChamferDistance
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

        dataloader = DataLoader(NPYDataset(filenames1, filenames2, args.folder2_rotations, args.folder2_flip_vert), batch_size=1, shuffle=False, num_workers=8, collate_fn=NPYDataset.collect_fn)

        chamfer_distance = ChamferDistance().cuda()
        chamfer_dist_all = []
        for pts1, pts2 in tqdm(dataloader):
            pts1 = pts1[:,:3].cuda()
            pts2 = pts2[:,:3].cuda()

            with torch.no_grad():
                cd = chamfer_distance(
                    pts1[None, ...].detach(),
                    pts2[None, ...],
                    bidirectional=True,
                    point_reduction='mean'
                    )

            #chamfer_dist_value = (cd_forward / pred_pcd.shape[0]) + (cd_backward / gt_pcd.shape[0])
            chamfer_dist_value = cd.item()
            chamfer_dist_value = chamfer_dist_value / 2.0
            chamfer_dist_all.append(chamfer_dist_value)
        
        hist, bin_edges = np.histogram(chamfer_dist_all, bins=100, range=(0, 5))
        print("每个bin的数量:", hist)
        print("bin边界:", bin_edges)

if __name__ == "__main__":
    main()
