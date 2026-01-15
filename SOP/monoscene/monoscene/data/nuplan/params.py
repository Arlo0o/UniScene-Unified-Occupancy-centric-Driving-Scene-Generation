import numpy as np
import pickle

nuplan_class_frequencies = np.array([
    1076609424912.0,
    24204107447.0,
    2725876657.0,
    5266394.0,
    310678116.0,
    21253288.0,
    13689787.0,
    1036629.0,
    211066770.0
])

nuplan_class_names = [
    "empty",
    "background",
    "vehicle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
]