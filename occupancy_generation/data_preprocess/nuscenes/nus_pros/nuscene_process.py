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
from refile import smart_makedirs, smart_path_join, smart_listdir, smart_exists, smart_open

def load_occ_gt(occ_path,  ## 原始分辨率800*800*64, 调整grid_size进行下采样
                grid_size=np.array((800, 800, 64)), # 200, 200, 16
                unoccupied=0,voxel_size= np.array([100/800, 100/800, 8/64])
                ):
    #  [z y x cls] or [z y x vx vy vz cls]
    # pcd = np.load( occ_path, encoding='bytes', allow_pickle=True)
    pcd = np.load(smart_open(occ_path, 'rb'), encoding='bytes', allow_pickle=True)
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

    noise_mask = processed_label == 255
    processed_label[noise_mask] = 0
    return processed_label
def voxel2world(voxel,
                voxel_size=np.array([0.5, 0.5, 0.5]),
                pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])):
    """
    voxel: [N, 3]
    """
    return voxel * voxel_size[None, :] + pc_range[:3][None, :]
def world2voxel(wolrd,
                voxel_size=np.array([0.5, 0.5, 0.5]),
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


def find_npy_files(directory):
    directory_path = Path(directory)
    npy_files = list(directory_path.rglob('*.npy'))
    return [str(file) for file in npy_files]



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
    orignial_size = labels.shape
    reshaped = labels.reshape((quantize_size[0], 
                                orignial_size[0]//quantize_size[0], 
                                quantize_size[1], 
                                orignial_size[1]//quantize_size[1], 
                                quantize_size[2], 
                                orignial_size[2]//quantize_size[2]))
    result = np.zeros(quantize_size, dtype=labels.dtype)
    for i in range(quantize_size[0]):
        for j in range(quantize_size[1]):
            for k in range(quantize_size[2]):
                block = reshaped[i, :, j, :, k, :]
                result[i, j, k] = get_max_occurrence_label(block.flatten())

    return result

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
                 occ_base_path,
                 save_base_path,
                 quantize_size=(1., 1., 1.),
                 method='max',
                 s_t=0,
                 e_t=200012
                 ):
        self.config_path = config_path
        self.occ_base_path = occ_base_path
        self.save_base_path = save_base_path
        self.quantize_size = quantize_size

        self.labels_map = None

        config_file = yaml.safe_load(open(self.config_path, 'r'))
        
        # self.eval_fine_list = find_npy_files(occ_base_path)
        self.eval_fine_list = smart_listdir(occ_base_path)[s_t:e_t]
        
        # self.eval_fine_list = []
        # self.scenes = [os.path.join(occ_base_path, subset, scene, 'cartesian') 
        #                for subset in ['Train', 'Val', 'Test'] 
        #                for scene in os.listdir(os.path.join(occ_base_path, subset))]
        # for scene in self.scenes:
        #     eval_fine_dir = os.path.join(scene, 'evaluation_fine')
        #     frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(eval_fine_dir)) if filename.endswith('.label')]
        #     self.eval_fine_list.extend([os.path.join(eval_fine_dir, str(frame).zfill(6)+'.label') for frame in frames_list])

        self.__process_config(config_file)
        self.process_and_save_labels(method)

    def __process_config(self, config_file):
        _labels_map = config_file['learning_map']
        self.labels_map = np.asarray(list(_labels_map.values()))

    def __get_eval_fine(self, label_path):
        eval_fine = np.load(label_path, allow_pickle=True)
        return eval_fine

    def process_and_save_labels(self, method):
        for sample_token in tqdm(self.eval_fine_list, desc="Processing labels"):
            # eval_label = self.__get_eval_fine(sample_token)
            
            quantized_dir = os.path.join(self.save_base_path, f'nuscene_quantized_{self.quantize_size[0]}_{self.quantize_size[1]}_{self.quantize_size[2]}/', 'quantized')
            # import pdb; pdb.set_trace()
            subpath = os.path.join( *sample_token.split('/')[-2:])
            subdir_path = os.path.join(quantized_dir, *subpath.split('/')[:-1])
            os.makedirs(subdir_path, exist_ok=True)
            save_path = os.path.join(quantized_dir, subpath )
            if os.path.exists(f"{save_path}.npy"):
                continue
            
            
            lidar_token = smart_listdir(smart_path_join(self.occ_base_path, sample_token))[0][:-4]
            occ_path = smart_path_join(self.occ_base_path, sample_token, lidar_token + '.npy')
            eval_label = load_occ_gt(occ_path) 
            # import pdb; pdb.set_trace()
            eval_label = self.labels_map[eval_label]
             

            if self.quantize_size == (256, 256, 16):
                resampled_label = eval_label
            else:
                if method == 'max':
                    # print(eval_label.shape, self.quantize_size)
                    
                    resampled_label = resample(eval_label, self.quantize_size)
                elif method == 'first':
                    resampled_label = resample_first(eval_label, self.quantize_size)
                elif method == 'value_max':
                    resampled_label = resample_value_max(eval_label, self.quantize_size)
            
            # relative_path = os.path.relpath(sample_token, self.occ_base_path)
            # scene_dir = os.path.dirname(relative_path)
            # print(save_path)
            np.save(save_path, resampled_label)

def main():
    parser = argparse.ArgumentParser(description="Quantize and resample labels.")
    parser.add_argument('--quantize_size', type=int, nargs=3, required=True,
                        help="Set the quantize size (format: x y z)")
    parser.add_argument('--occ_base_path', type=str, default='../../data/Cartesian',
                        help="Set the base path for data (default: '../../data/Cartesian')")
    parser.add_argument('--save_base_path', type=str, default='../../data',
                        help="Set the base path for saving output (default: '../../data')")
    parser.add_argument('--config_path', type=str, default='carla.yaml',
                        help="Set the config file path (default: 'carla.yaml')")
    parser.add_argument('--method', type=str, default='max', choices=['max', 'first', 'value_max'],
                        help="Set the resampling method (default: 'max')")
    parser.add_argument('--s_e', type=int, nargs=2, required=True,
                        help="Set the start and end list (format: s e)")

    args = parser.parse_args()

    Quantize(config_path=args.config_path, 
             occ_base_path=args.occ_base_path,
             save_base_path=args.save_base_path,
             quantize_size=tuple(args.quantize_size),
             method=args.method,
             s_t=args.s_e[0],
             e_t=args.s_e[1],
             )
    
if __name__ == '__main__':
    main()

    # python nus_pros/nuscene_process.py \
    # --quantize_size  200 200 16  \
    # --occ_base_path "s3://sdagent-shard-bj-baiducloud/crosshairs/zouyingshuang-share/occ/8-30-nksr/dense_voxels_with_semantic/" \
    # --save_base_path "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/occ_12hz" \
    # --config_path "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/code/occworld-dev_12hz/nus_pros/nuscene.yaml" \
    # --method "max"