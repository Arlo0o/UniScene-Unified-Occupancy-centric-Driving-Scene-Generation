from collections import defaultdict
import numpy as np
VELODYNE_HDL32E_ELEVATION_MAPPING = dict(zip(np.arange(32), tuple(np.linspace(-30.67, 10.67, 32))))

NUPLAN_LIDAR_NUM = 5
MAX_RELECTANCE_VALUE = 255.0
LIDAR_FREQUENCY = 20.0  # Hz
LIDAR_CHANNELS = 40  # number of vertical channels

NUPLAN_LIDAR_LOCS_NAIVE = {
            0: [0, 0, 0],
            1: [0, 0, 0],
            2: [0, 0, 0],
            3: [0, 0, 0],
            4: [0, 0, 0],
        }

NUPLAN_LIDAR_LOCS = {
            0: [ 1.5283, -0.0081,  1.6157],
            1: [2.5804, 0.8384, 1.0807],
            2: [ 2.5725, -0.9060,  1.0997],
            3: [-1.0422, -0.2772,  0.6041],
            4: [ 4.0223, -0.0238,  0.0084],
        }

NUPLAN_LIDAR_RAY_RANGE = {
    0: [-50, -0.0001, 1.6157, -50, 0.0001, 1.6157, 360],
    1: [-8.68, -49.24, 1.0807, -50, 0.0001, 1.0807, 280],
    2: [-50, -0.0001, 1.0997, -8.68, 49.24, 1.0997, 280],
    3: [-17.10, 46.98, 0.6041, -8.68, -49.24, 0.6041, 150],
    4: [0, -50, 0.0084, 0, 50, 0.0084, 180]
}

NUPLAN_THETA_MAP = {
    0: [-25, -19, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5.66, -5.33, -5, -4.66, -4.33, -4, -3.66, -3.33, -3, -2.66, -2.33, -2, -1.66, -1.33, -1, -0.66, -0.33, 0, 0.33, 0.66, 1, 1.33, 1.66, 2, 3, 5, 8, 11, 15],
    1: [-25, -19, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5.66, -5.33, -5, -4.66, -4.33, -4, -3.66, -3.33, -3, -2.66, -2.33, -2, -1.66, -1.33, -1, -0.66, -0.33, 0, 0.33, 0.66, 1, 1.33, 1.66, 2, 3, 5, 8, 11, 15],
    2: [-25, -19, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5.66, -5.33, -5, -4.66, -4.33, -4, -3.66, -3.33, -3, -2.66, -2.33, -2, -1.66, -1.33, -1, -0.66, -0.33, 0, 0.33, 0.66, 1, 1.33, 1.66, 2, 3, 5, 8, 11, 15],
    3: [-19, -14, -12, -10, -8, -7, -6, -5, -4, -3, -2.33, -1.67, -1, -0.33, 0, 0.33, 1, 1.67, 2, 3],
    4: [-25, -19, -14, -12, -10, -8, -6, -5, -4, -3, -2, -1, -0.33, 0.33, 1, 1.67, 2, 3, 5, 8],
    #4: [-19, -14, -12, -10, -8, -7, -6, -5, -4, -3, -2.33, -1.67, -1, -0.33, 0, 0.33, 1, 1.67, 2, 3],
}

NUPLAN_UNIQUE_LIDAR_IDS = [0, 1, 2, 3, 4]

# Nuscenes defines actor coordinate system as x-forward, y-left, z-up
# But we want to use x-right, y-forward, z-up
# So we need to rotate the actor coordinate system by 90 degrees around z-axis
WLH_TO_LWH = np.array(
    [
        [0, 1.0, 0, 0],
        [-1.0, 0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0],
    ]
)
HORIZONTAL_BEAM_DIVERGENCE = 0.00333333333  # radians, given as 4 inches at 100 feet
VERTICAL_BEAM_DIVERGENCE = 0.00166666666  # radians, given as 2 inches at 100 feet

NUSCENES_ELEVATION_MAPPING = {
    "LIDAR_TOP": VELODYNE_HDL32E_ELEVATION_MAPPING,
}
NUSCENES_AZIMUTH_RESOLUTION = {
    "LIDAR_TOP": 1 / 3.0,
}
NUSCENES_SKIP_ELEVATION_CHANNELS = {
    "LIDAR_TOP": (
        0,
        1,
    )
}
AVAILABLE_CAMERAS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)


DEFAULT_IMAGE_HEIGHT = 1080
DEFAULT_IMAGE_WIDTH = 1920

DUMMY_DISTANCE_VALUE = 2e3


# CAM_FRONT_INTRINSIC = np.array([[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],
#        [0.00000000e+00, 1.26641720e+03, 4.91507066e+02],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


# CAM_FRONT_EXTRINSIC = np.array([[ 0.9999401 , -0.00841551, -0.0069986 ],
#                             [ 0.00745669,  0.05569509,  0.99841998],
#                             [-0.00801243, -0.99841236,  0.05575451]])