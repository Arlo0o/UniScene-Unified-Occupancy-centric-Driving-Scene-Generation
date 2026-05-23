from collections import defaultdict
import numpy as np
VELODYNE_HDL32E_ELEVATION_MAPPING = dict(zip(np.arange(32), tuple(np.linspace(-30.67, 10.67, 32))))

MAX_RELECTANCE_VALUE = 255.0
LIDAR_FREQUENCY = 20.0  # Hz
LIDAR_CHANNELS = 32  # number of vertical channels
ALLOWED_RIGID_CLASSES = (
    "vehicle.car",
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "vehicle.bus",
    "vehicle.bus",
    "vehicle.truck",
    "vehicle.trailer",
    "movable_object.pushable_pullable",
)
ALLOWED_DEFORMABLE_CLASSES = ("human.pedestrian",)

TRACKING_TO_GT_CLASSNAME_MAPPING = {
    "pedestrian": "human.pedestrian",
    "bicycle": "vehicle.bicycle",
    "motorcycle": "vehicle.motorcycle",
    "car": "vehicle.car",
    "bus": "vehicle.bus",
    "truck": "vehicle.truck",
    "trailer": "vehicle.truck",
}
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
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "BACK",
    "BACK_LEFT",
    "BACK_RIGHT",
)
CAMERA_TO_BOTTOM_RIGHT_CROP = {
    "FRONT": (0, 0),
    "FRONT_LEFT": (0, 0),
    "FRONT_RIGHT": (0, 0),
    "BACK": (80, 0),
    "BACK_LEFT": (0, 0),
    "BACK_RIGHT": (0, 0),
}
SEQ_CAMERA_TO_BOTTOM_RIGHT_CROP = defaultdict(lambda: CAMERA_TO_BOTTOM_RIGHT_CROP)
SEQ_CAMERA_TO_BOTTOM_RIGHT_CROP["scene-0164"] = {
    "FRONT": (0, 0),
    "FRONT_LEFT": (0, 0),
    "FRONT_RIGHT": (0, 0),
    "BACK": (80 + 66, 0),
    "BACK_LEFT": (0, 0),
    "BACK_RIGHT": (0, 0),
}

DEFAULT_IMAGE_HEIGHT = 900
DEFAULT_IMAGE_WIDTH = 1600

DUMMY_DISTANCE_VALUE = 2e3


CAM_FRONT_INTRINSIC = np.array([[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],
       [0.00000000e+00, 1.26641720e+03, 4.91507066e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


CAM_FRONT_EXTRINSIC = np.array([[ 0.9999401 , -0.00841551, -0.0069986 ],
                            [ 0.00745669,  0.05569509,  0.99841998],
                            [-0.00801243, -0.99841236,  0.05575451]])