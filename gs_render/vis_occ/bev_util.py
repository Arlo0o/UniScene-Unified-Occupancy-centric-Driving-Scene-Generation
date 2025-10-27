
import numpy as np
import cv2

def replace_occ_grid_with_bev_nuplan(input_occ, bevlayout, driva_area_idx=1, bev_replace_idx=[1,15, 17],
                                     occ_replace_new_idx=[12, 14, 15]):
    # self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
    # nuplan
    # self.classes= ['intersections','generic_drivable_areas','walkways','carpark_areas','crosswalks','lane_group_connectors','lane_groups_polygons','road_segments']
    # 需要把 walkways 换成另外一种颜色,10
    # 需要把 drivable_area 换成另外一种颜色,11
    # lane_divider 换成另外一种颜色,12
    # stop_line del
    # road_block 13
    # occ road [11] drivable area

    # default ped_crossing->18; stop_line->19 (del); roal_divider->20; lane_divider->21
    # default shape: input_occ: [200,200,16]; bevlayout: [18,200,200]

    roal_divider_mask = bevlayout[15, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[17, :, :].astype(np.uint8)

    # roal_divider_mask = cv2.dilate(
    #     roal_divider_mask, np.ones((3, 3), np.uint8))
    # lane_divider_mask = cv2.dilate(
    #     lane_divider_mask, np.ones((3, 3), np.uint8))

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