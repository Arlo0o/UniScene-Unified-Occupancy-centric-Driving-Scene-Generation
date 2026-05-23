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


tokens = ['38afa74ed5be5c81', '858ded2dfafe52a3', '4e9e1eb8fbbf5ee9', 'bd486a7c1c265a2b', '937366bd50ac5af5', 'b66a5a456377585a', '3550ab13d47f532e', 'ebb87cff42d751ee', '3527fec459b556fb', '0357ea2c63d85ce7', '83c7d17f91085977', 'cbe0f54641945253', 'cde691b9d1e850c6', '37aa0dcc30d6545b', '92c6df3afe9057c6', '07a1b47e5f3c59a8', '565d186dd571542f', '64ab3e5cf5b55d1b', '3a977e2bb9d659d6', '5b743b1db1a15839', 'facba09ebf535674', 'e6907ba3cca85c2a', 'f52b0c3bd71450e1', '84c64452b4f35dea', '781ce25aed625cd3', '87e425d5b12153de', 'd5aa50fa0fa15295', 'be9ce54edae350a9', '2d0db2914e5052d5', '3a11554fdf9f54b1', '92c8bedbf6e15a3f', 'cab2ee15ca20543a', 'a0a41cfc933856d5', '77d4cb4063d35521', '27c0045dcc535ee0', '1929d45ae0405558', 'aea453fd69e158c4', '0de00743162953d0', '579a8b2b1e695fff', '549e2f29d2375533', '4ccdd96a25d25cf8', 'f0a343a472e4572e', 'f722b78446ae5690', 'a3e1939046d05111', '2e3bf465d8425a76', 'c776287be5615a73', 'b2208ea2274751a7', '499c9dbbdbd5501b', 'ebcf1cdd19ca58d3', '244ce0482fc159fb', '6b34571c519b52ca', 'ac0bd8d05b5c5a71', 'ebfe8a88fdd6587b', 'd1af4b200e6d5f07', '9ccd0a2551ff59fc', '484e76ea20295b1f', '1464b5445e2e5698', 'c376f4372a6e5bac', 'f5aa615c19dd528e', '90d65a2154e55218', 'd44aa98f9b855b16', 'ccfc2c6db88d588a', 'bf10014d411152a9', '8d6692404a7555fa', '1e10c1b29bb35c8d', '05fcd8cf06c45bc1', 'ab16ec3e141b569a', 'cf07c04676c75f79', '580f4798b57f5d5e', 'c93e95e85570558e', '2185b7467d205b1f', 'bc690c89c6165b4c', 'f1de188dc16e59a3', 'ad97e40342ce5b58', '523145189d115971', '3a762d778b59587f', '8ed2ae4685de5954', 'a670986a8b8e5ab9', '316665c646185dd9', 'da6a85740e9f5407', 'd3fb5d1945295316', 'c4aecb8832e554f3', '0daec8aa647b566f', 'bb689607824b513c', 'ad5b6e541be152e7', '4f71f41c302351d7', '2fbec6b63db65c84', '24ae8f3a418b5ca2', '00009841879a5bb9', 'd7b86f9643b35aac', 'b6711298938059a1', '369035069c855b9e', '4a5f50600dcf570c', 'e4518f7671815bc1', 'b20925347dad549e', '627c821610af54d3', 'fb5867fdadca5886', 'b610ae1f3532563b', '0a52a24f4df55651', '9bd1f1ef99ae5fd5', 'eeb1174af7c75788', 'ab3d999e4a205f43', '2f8267c125fc566a', '717b6919d966523c', 'f66b230d3a73593a', 'f30f55ae2c2d5aa2', '242f4cc771bc52d8', '40fe3b156c1c526b', '95287235c86b5a45', 'e89086faf6b657a3', '3cd69f0f51d55500', '0b3560df91eb54c8', '0339e18bfaf05885', '229d4c0ae5985c22', '2090d5168c1a5c70', '9ccb6997ac545bdc', 'c946c1f66aba58dc', '546a27bfdb5a571a', '50a85e2ca4c65063', 'b52e75f4af395fcf', 'c14db22603de54d6', '14018473672f501c', '9fc37b54fc21545e', '001f128eb0215209', 'b448af2b0da75298', '018713bae1a354f3', '51e46451f0c75eff', '726f1316f24753a6', '21a8479baeae5b05', 'be5735bf313a55f3', 'dca14cebc065511c', '89ab083ffeb75134', '8fd32cd2fa1e5498', 'f8b8dd28958a5ba4', '2feb5eee51d453f9', '9c667fdb5bbe5c2a', '928f3f5734445127', '8f26e4dff6285eb0', '4f2fab8c2ba25b9c', '3db828487c9a5c22', '330ab78f09ab5a0c', 'aa2c3aa0650f5cba', 'f20c9f0bfd8254a3', 'b976de348bc15d25', 'c985a4d00eaa50c4', '47ab205b713f5e74', 'c15c9f0f1786560c', '524827f4c8fc5147', '0c6a0d5e863251c3', '3884ecf9935b57fd', '275d1391512a5451', 'c9a01c5e9d32548d', 'a63732e3e31655b3', '7c901feab391576a', 'c5459519695d5306', 'f27d7ede60015e54', '02c0c817ca905bb7', '8c8a0f6b46c256d0', 'f53467bbb8455af8', 'b8e66733e4b2529f', 'a8cf6d484da75d66', 'c207fbf190765eb9', '9e4e95f91ab45b2f', 'c5f9a838899a535a', '86199b8759a65f07', 'b27a7e12fdb35bef', '8fc11fdac1a759cc', 'dc31ba530f485dd2', 'dd19b9206a3355b7', '635e46c255cf5f11', 'c9b4601027805dee', '15f4a482c38e5c8a', '42bc66613a335451', 'fa8fefb27ea8505d', '1637b00ad1365152', '017c98c16dc05a36', '8b2a3dd64e485c56', '4171de35ea7357ca', 'c9003d17041159a9', '304e601281a45b22', 'bee569151aef59bf', 'dd49421958fa5259', '3231f31c48735ca9', '31b559fe3db157c6', 'd66c62a0654c5dea', '894f853a769f5ad9', '6d947839c6245440', 'a8919134c4325d82', '88355536492a5def', '112fd3c1349c59f0', 'fce72f72b6b156e9', 'f6b530a892405be9', 'f8e630fcce3a522a', '8196e821064c54c3', '41ced78125d85845', 'a084a52dca7e53ce', '9537b215e16f55fe', 'e678b03a82be56a7', '08eafd137c4e57fc', 'd336f7eaf44853e6', '9a19996dbecd5523']

def find_npy_files(directory):
    # directory_path = Path(directory)
    npz_files = []
    for token in tokens:
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
        # self.data_infos = data['infos']
        self.config_path = config_path
        self.data_base_path = data_base_path
        self.save_base_path = save_base_path
        self.quantize_size = quantize_size

        self.labels_map = None

        config_file = yaml.safe_load(open(self.config_path, 'r'))
        
        self.eval_fine_list = find_npy_files(data_base_path)
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

