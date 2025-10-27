
import numpy as np
import cv2

def replace_occ_grid_with_bev_nuplan(input_occ, bevlayout, driva_area_idx=1, bev_replace_idx=[1,15, 17],
                                     occ_replace_new_idx=[12, 14, 15]):

    roal_divider_mask = bevlayout[15, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[17, :, :].astype(np.uint8)

    roal_divider_mask = cv2.dilate(
        roal_divider_mask, np.ones((2, 2), np.uint8))
    lane_divider_mask = cv2.dilate(
        lane_divider_mask, np.ones((2, 2), np.uint8))

    bevlayout[15, :, :] = roal_divider_mask.astype(bool)
    bevlayout[17, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ.shape[0], input_occ.shape[1]
    output_occ = input_occ.copy()  # numpy copy() ; tensor clone()
    bev_replace_mask = []
    for i in range(n):
        bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)

    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ[x, y, :]

                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(
                            occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
    return output_occ

def add_bev_layout(occ_path, bev_path):
    occ = np.load( occ_path, encoding='bytes', allow_pickle=True)['occ']
    
    # occ with bev
    bev_data = np.load(bev_path)['gt_bev_masks']
    bev = bev_data.copy()
    layer_to_merge=[0,1,3,4,5,6,7]
    bev[1,:,:] = np.any(bev[layer_to_merge, :, :], axis=0).astype(int)
    grid = np.zeros((400, 400, 32), dtype=np.uint8)
    grid[occ[:, 0], occ[:, 1], occ[:, 2]] = occ[:, 3]+1
    occ_with_bev = replace_occ_grid_with_bev_nuplan(grid, bev, bev_replace_idx=[1,15],occ_replace_new_idx=[12,14])
    _loc = np.stack(occ_with_bev.nonzero(), axis=-1)
    _label = occ_with_bev[_loc[:, 0], _loc[:, 1], _loc[:, 2]]
    occ_with_bev = np.concatenate([_loc, _label[:, None]], axis=-1)
    return occ_with_bev