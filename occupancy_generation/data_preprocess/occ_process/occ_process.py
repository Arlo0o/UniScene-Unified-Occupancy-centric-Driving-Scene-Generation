import numpy as np
from scipy.ndimage import zoom
import yaml
from numba import njit
import os
from tqdm import tqdm
import dask.array as da
import argparse
from pathlib import Path
import numpy as np
import numba as nb
import copy
from scipy.ndimage import median_filter
import math
import pickle

# 定义类别权重字典，给小目标更高权重，防止采样时丢失
weight_dict = {
    0: 1,   # free 
    1: 10,   # vehicle
    2: 30,   # pedestrian
    3: 10,   # traffic_cone
    4: 8,   # barrier
    5: 10,  # czone_sign
    6: 8,   # generic_object
    7: 4,   # background
    8: 2    # unknown
}
# 将 weight_dict 转换成 NumPy 数组，Numba 不支持直接用 dict
weight_dict_keys = np.array(list(weight_dict.keys()), dtype=np.int32)
weight_dict_values = np.array(list(weight_dict.values()), dtype=np.float32)

def convert_to_voxel_grid_int(voxels_):
    voxel = np.zeros((800, 800, 64), dtype=np.int32)
    voxel[voxels_[:, 0].astype(np.int32), voxels_[:, 1].astype(np.int32), voxels_[:, 2].astype(np.int32)] = voxels_[:, 3].astype(np.int32)+1
    return voxel


def load_occ_gt(occ_path,  ## 原始分辨率800*800*64, 调整grid_size进行下采样
                grid_size=np.array((800, 800, 64)), # 200, 200, 16
                unoccupied=0,voxel_size= np.array([100/800, 100/800, 8/64])
                ):
    #  [z y x cls] or [z y x vx vy vz cls]
    # pcd = np.load(occ_path, encoding='bytes', allow_pickle=True)
    pcd = np.load(occ_path)['occ']
    pcd[:,3] = pcd[:,3]+1
    print(f"pcd shape: {pcd.shape}")
    for i in range(20):
        if np.any(pcd[:, 3] == i):
            print(f"label {i} in points_with_label {np.any(pcd[:, 3] == i)} sum {np.sum(pcd[:, 3] == i)}")

    pcd_label = pcd[..., -1:]
    pcd_label[pcd_label == 0] = 255
    pcd_np_cor = voxel2world(pcd[..., [0, 1, 2]] + 0.5, voxel_size=voxel_size )  # x y z
    untransformed_occ = copy.deepcopy(pcd_np_cor)  # N 4

    # bevdet augmentation
    # pcd_np_cor = (results['bda_mat'] @ torch.from_numpy(pcd_np_cor).unsqueeze(-1).float()).squeeze(-1).numpy()
    pcd_np_cor = world2voxel(pcd_np_cor, voxel_size=voxel_size)

    # make sure the point is in the grid
    pcd_np_cor = np.clip(pcd_np_cor, np.array([0, 0, 0]), grid_size - 1)
    transformed_occ = copy.deepcopy(pcd_np_cor)
    pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

    # 255: noise, 1-16 normal classes, 0 unoccupied
    pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
    pcd_np = pcd_np.astype(np.int64)
    processed_label = np.ones(grid_size, dtype=np.uint8) * unoccupied
    processed_label = nb_process_label(processed_label, pcd_np)
    # processed_label = convert_to_voxel_grid_int(pcd)
    print(f"processed_label shape: {processed_label.shape}")

    noise_mask = processed_label == 255
    processed_label[noise_mask] = 0
    print(f"processed_label shape: {processed_label.shape}")
    return processed_label

def voxel2world(voxel,
                voxel_size=np.array([0.125, 0.125, 0.125]),
                pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])):
    """
    voxel: [N, 3]
    """
    return voxel * voxel_size[None, :] + pc_range[:3][None, :]
def world2voxel(wolrd,
                voxel_size=np.array([0.125, 0.125, 0.125]),
                pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])):
    """
    wolrd: [N, 3]
    """
    return (wolrd - pc_range[:3][None, :]) / voxel_size[None, :]

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def find_npy_files(directory, data_infos):
    # directory_path = Path(directory)
    npz_files = []
    for info in data_infos:
        token = info['token']
        npz_file =  f"{token}/{token}.npz"
        npz_files.append(os.path.join(directory, npz_file))
    return npz_files

@njit
def get_max_occurrence_label(block):
    counts = np.bincount(block)
    nonzero_counts = counts[1:]
    if nonzero_counts.size > 0:
        max_label = np.argmax(nonzero_counts) + 1
        return max_label
    else:
        return 0

@njit
def resample(labels, quantize_size):
    labels = np.ascontiguousarray(labels)
    original_size = labels.shape
    reshaped = labels.reshape((quantize_size[0], 
                               original_size[0]//quantize_size[0], 
                               quantize_size[1], 
                               original_size[1]//quantize_size[1], 
                               quantize_size[2], 
                               original_size[2]//quantize_size[2]))
    result = np.zeros(quantize_size, dtype=labels.dtype)
    for i in range(quantize_size[0]):
        for j in range(quantize_size[1]):
            for k in range(quantize_size[2]):
                block = reshaped[i, :, j, :, k, :]
                result[i, j, k] = get_max_occurrence_label(block.flatten())
    return result

def split_and_resample(labels, original_z_split, target_size_below, target_size_above):
    print(f"=split ground ")
    # 分割地面以下和地面以上部分
    ground_below = labels[:, :, :original_z_split]
    ground_above = labels[:, :, original_z_split:]
    
    # 分别降采样
    resampled_below = resample(ground_below, (target_size_below[0], target_size_below[1], target_size_below[2]))
    resampled_above = resample(ground_above, (target_size_above[0], target_size_above[1], target_size_above[2]))
    
    # 合并结果
    final_result = np.concatenate((resampled_below, resampled_above), axis=2)
    return final_result


def resample_first(labels, scale_factors):
    scale_factors = scale_factors / np.asarray(labels.shape)
    labels_resampled = zoom(labels, scale_factors, order=0, mode='nearest', prefilter=False)
    for index, _ in np.ndenumerate(labels_resampled):
        original_index = tuple(slice(int(i / scale), int((i + 1) / scale)) for i, scale in zip(index, scale_factors))
        original_block = labels[original_index]
        labels_resampled[index] = next((x for x in original_block.flat if x), 0)
    return labels_resampled

def resample_value_max(labels, scale_factors):
    shape = np.array(labels.shape)
    reduction_factor = tuple(old_dim // new_dim for old_dim, new_dim in zip(shape, scale_factors))

    reduction_dict = {i: factor for i, factor in enumerate(reduction_factor)}

    labels_dask = da.from_array(labels)

    labels_resampled = da.coarsen(np.max, labels_dask, reduction_dict)

    return labels_resampled.compute()

class Quantize:
    def __init__(self, 
                 config_path,
                 data_base_path,
                 save_base_path,
                 quantize_size=(1., 1., 1.),
                 method='max',
                 ):
        pkl_file_path = "/mnt/dataset/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_train.pkl"
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        self.data_infos = data['infos']
        self.config_path = config_path
        self.data_base_path = data_base_path
        self.save_base_path = save_base_path
        self.quantize_size = quantize_size

        self.labels_map = None

        config_file = yaml.safe_load(open(self.config_path, 'r'))
        
        self.eval_fine_list = find_npy_files(data_base_path, self.data_infos)
        print(f"length of eval_fine_list: {len(self.eval_fine_list)}")

        self.__process_config(config_file)
        self.process_and_save_labels(method)

    def __process_config(self, config_file):
        _labels_map = config_file['learning_map']
        print(f"_labels_map: {_labels_map}")
        self.labels_map = np.asarray(list(_labels_map.values()))
        print(f"self.labels_map: {self.labels_map}")

    def __get_eval_fine(self, label_path):
        eval_fine = np.load(label_path, allow_pickle=True)
        return eval_fine

    def process_and_save_labels(self, method):
        for label_path in tqdm(self.eval_fine_list, desc="Processing labels"):
            # eval_label = self.__get_eval_fine(label_path)
            
            eval_label = load_occ_gt(label_path) 
            print(f"eval_label shape: {eval_label.shape}==")
            # import pdb; pdb.set_trace()

            eval_label = self.labels_map[eval_label]
             

            if self.quantize_size == (256, 256, 16):
                resampled_label = eval_label
            else:
                if method == 'max':
                    print(f"resampling with max method, and the eval_label shape is {eval_label.shape} quantize size is {self.quantize_size}")
                    original_z_split = 40
                    if self.quantize_size[2] == 32:
                        target_size_below = (400, 400, 8)
                        target_size_above = (400, 400, 24)
                    elif self.quantize_size[2] == 16:
                        target_size_below = (200, 200, 4)
                        target_size_above = (200, 200, 12)

                    # resampled_label = split_and_resample(eval_label, original_z_split, target_size_below, target_size_above)
                    resampled_label = resample(eval_label, self.quantize_size)                    

                elif method == 'first':
                    resampled_label = resample_first(eval_label, self.quantize_size)
                elif method == 'value_max':
                    resampled_label = resample_value_max(eval_label, self.quantize_size)
            
            # relative_path = os.path.relpath(label_path, self.data_base_path)
            # scene_dir = os.path.dirname(relative_path)
            quantized_dir = os.path.join(self.save_base_path, f'nuscene_quantized_{self.quantize_size[0]}_{self.quantize_size[1]}_{self.quantize_size[2]}/')
            
            # if not os.path.exists(quantized_dir):
            #     os.makedirs(quantized_dir)
            
            # import pdb; pdb.set_trace()
            subpath = os.path.join(*label_path.split('/')[-1:])
            # Remove file extension (.npz) from the filename
            subpath = os.path.splitext(subpath)[0]
            print(f"subpath: {subpath}")
 
            subdir_path = os.path.join(quantized_dir, *subpath.split('/')[:-1])
            os.makedirs(subdir_path, exist_ok=True)

            save_path = os.path.join(quantized_dir, subpath)
            print(f"save_path :{save_path} and ==> {resampled_label.shape}")
            np.save(save_path, resampled_label)

def main():
    parser = argparse.ArgumentParser(description="Quantize and resample labels.")
    parser.add_argument('--quantize_size', type=int, nargs=3, required=True,
                        help="Set the quantize size (format: x y z)")
    parser.add_argument('--data_base_path', type=str, default='../../data/Cartesian',
                        help="Set the base path for data (default: '../../data/Cartesian')")
    parser.add_argument('--save_base_path', type=str, default='../../data',
                        help="Set the base path for saving output (default: '../../data')")
    parser.add_argument('--config_path', type=str, default='carla.yaml',
                        help="Set the config file path (default: 'carla.yaml')")
    parser.add_argument('--method', type=str, default='max', choices=['max', 'first', 'value_max'],
                        help="Set the resampling method (default: 'max')")

    args = parser.parse_args()

    Quantize(config_path=args.config_path, 
             data_base_path=args.data_base_path,
             save_base_path=args.save_base_path,
             quantize_size=tuple(args.quantize_size),
             method=args.method
             )
    
if __name__ == '__main__':
    # _change_builtin_print(True)
    main()
    
    # python3 Tools/data_process/nuscene_process.py \
    # --quantize_size  100 100 8  \
    # --data_base_path "/data/longhun/3D/nuscenes/data/nksr_occ_fix" \
    # --save_base_path "/data/longhun/3D/nuscenes/data/pyramid_occ/" \
    # --config_path "/code/dis_scene/drive_scene/occnet/pyramid-discrete-diffusion/Tools/data_process/nuscene.yaml" \
    # --method "max"

