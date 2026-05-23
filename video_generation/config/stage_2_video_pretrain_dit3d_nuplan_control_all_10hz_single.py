from copy import deepcopy
val = True
num_cond_frame = 3
grad_checkpoint = True
num_frames = None
micro_frame_size = None#8
bbox_mode = None#'all-xyz'
template = "A driving scene video at {description}."
nuplan=True
drop_cond_ratio=0.1
object_classes = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

map_classes = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "road_divider",
    "lane_divider",
    "road_block",
]

input_modality = dict(
    use_lidar = False,
    use_camera = True,        
    use_radar = False,
    use_map = False,
    use_external = False,
)

img_collate_param_train = dict(
    # template added by code.
    frame_emb = "next2top",
    bbox_mode = bbox_mode,
    bbox_view_shared = False,
    keyframe_rate = 6,  # work with `bbox_drop_ratio`
    bbox_drop_ratio = 0.4,
    bbox_add_ratio = 0.1,
    bbox_add_num = 3,
    bbox_processor_type = 2,
    template=template
)

balance_keywords = None #["night", "rain", "none"]
scale_3d = [1.0, 1.0]  # adjust the scale
rotate_3d = [0.0, 0.0]  # rotation the lidar
translate_3d =  0  # shift
flip_ratio_3d = 0.0
flip_direction_3d = "null"

collect_meta_keys = [
    "camera_intrinsics",
    "lidar2ego",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
    "next2top"
]  # send to DataContainer
collect_meta_lis_keys = [
    # "timeofday",
    # "location",
    "description",
    "filename",
    "token",
] # hold by one DataContainer
 


view_order = [
    'CAM_L1',
    "CAM_L0",
    "CAM_F0",
    "CAM_R0",
    
    "CAM_R1",
    'CAM_R2',
    "CAM_B0",
    "CAM_L2",
]
mv_order_map = {
    0: [7, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 6],
    6: [5, 7],
    7: [6, 0],
    }



pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=True,
        img_path = "./dataset1/nuplan/sensor_blobs_train/",
        seg_path ="dataset1/nuplan-occ-render-mini/mini", 
        depth_path = "dataset1/nuplan-occ-render-mini/mini", 
        occ_path =  None # "/lpai/dataset/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16/",
        ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d = True,
        with_label_3d = True,
        with_attr_label = False
    ),
    dict(
        type="ImageAug3D",
        final_dim=[224, 400],
        resize_lim=[0.25, 0.25],
        bot_pct_lim = [0.0, 0.0],
        rot_lim='null',
        rand_flip=False,
        is_train=False
    ),
    dict(
        type="GlobalRotScaleTrans",
        resize_lim=scale_3d,
        rot_lim=rotate_3d,
        trans_lim=translate_3d,
        is_train=True
    ),
    dict(
        type="ObjectNameFilter",
        classes=object_classes
    ),
    dict(
        type="ReorderMultiViewImages",
        order=view_order,
        safe=False,
        dataset = "nuplan" ,
    ),
    dict(
        type="ImageNormalize",
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5]
    ),
    dict(
        type="DefaultFormatBundle3D",
        classes=object_classes
    ),
    dict(
        type="Collect3D",
        keys=[
            "img",
            # "lidar",
            "semantic_map",
            "depth_map",
            # "occ",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "img_aug_matrix",
        ],
        meta_keys=collect_meta_keys,
        meta_lis_keys=collect_meta_lis_keys,
    ),
]

pipeline2 = deepcopy(pipeline)
pipeline2[2] = dict(
    type="ImageAug3D",
    final_dim=[448, 800],
    resize_lim=[0.5, 0.5],
    bot_pct_lim = [0.0, 0.0],
    rot_lim='null',
    rand_flip=False,
    is_train=False
)

pipeline3 = deepcopy(pipeline)
pipeline3[2] = dict(
    type="ImageAug3D",
    final_dim=[112, 192],
    resize_lim=[0.125, 0.125],
    bot_pct_lim = [0.0, 0.0],
    rot_lim='null',
    rand_flip=False,
    is_train=False
)



dataset_cfg_list = [
    ((224, 400),dict(
        type = "NuplanVariableDataset",
        ann_file=["./pickle/nuplan_mini_10hz_pkl/nuplan_mini_train.pkl", 
                #   "./pickle/nuplan_mini_10hz_pkl/nuplan_mini_val.pkl", 
                #   "./pickle/nuplan_10hz_pkl/trainval/nuplan_trainval_10hz_val.pkl" 
                  ],
        pipeline=pipeline,
        dataset_root=None,
        object_classes=object_classes,
        map_classes=map_classes,
        load_interval=1,
        with_velocity=True,
        modality=input_modality,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length =  [ 17, 33, 65, 129, ], #[1,17,"full"],#[1, 17, 33, 65, "full"],
        start_on_keyframe=False,
        next2topv2=True,
        trans_box2top=False,
        base_fps=10,
        fps = [  [10], [10], [10], [10],  ],  
        # repeat_times = [1],#[1, 1, 40],
        img_collate_param=img_collate_param_train,
        micro_frame_size=micro_frame_size,
        balance_keywords=balance_keywords,
        drop_ori_imgs=False,
    )),
    
      ((448, 800),dict(
        type = "NuplanVariableDataset",
        ann_file=["./pickle/nuplan_mini_10hz_pkl/nuplan_mini_train.pkl", 
                #   "./pickle/nuplan_mini_10hz_pkl/nuplan_mini_val.pkl", 
                #   "./pickle/nuplan_10hz_pkl/trainval/nuplan_trainval_10hz_val.pkl" 
                  ],
        pipeline=pipeline2,
        dataset_root=None,
        object_classes=object_classes,
        map_classes=map_classes,
        load_interval=1,
        with_velocity=True,
        modality=input_modality,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length =  [ 9, 17 ], #[1,17,"full"],#[1, 17, 33, 65, "full"],
        start_on_keyframe=False,
        next2topv2=True,
        trans_box2top=False,
        base_fps=10,
        fps = [  [10], [10]  ],  
        # repeat_times = [1],#[1, 1, 40],
        img_collate_param=img_collate_param_train,
        micro_frame_size=micro_frame_size,
        balance_keywords=balance_keywords,
        drop_ori_imgs=False,
    )),
      
    ((112, 192),dict(
        type = "NuplanVariableDataset",
        ann_file=["./pickle/nuplan_mini_10hz_pkl/nuplan_mini_train.pkl", 
                #   "./pickle/nuplan_mini_10hz_pkl/nuplan_mini_val.pkl", 
                #   "./pickle/nuplan_10hz_pkl/trainval/nuplan_trainval_10hz_val.pkl" 
                  ],
        pipeline=pipeline3,
        dataset_root=None,
        object_classes=object_classes,
        map_classes=map_classes,
        load_interval=1,
        with_velocity=True,
        modality=input_modality,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length =  [ 129, 201  ], #[1,17,"full"],#[1, 17, 33, 65, "full"],
        start_on_keyframe=False,
        next2topv2=True,
        trans_box2top=False,
        base_fps=10,
        fps = [  [10], [10], ],  
        # repeat_times = [1],#[1, 1, 40],
        img_collate_param=img_collate_param_train,
        micro_frame_size=micro_frame_size,
        balance_keywords=balance_keywords,
        drop_ori_imgs=False,
    )),
    
]

val_dataset_cfg_list = [
    ((224, 400),dict(
        type = "NuplanVariableDataset",
        ann_file=["./pickle/nuplan_mini_10hz_pkl/nuplan_mini_train.pkl"],
        pipeline=pipeline,
        dataset_root=None,
        object_classes=object_classes,
        map_classes=map_classes,
        load_interval=1,
        with_velocity=True,
        modality=input_modality,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length = [ 33, 65, 129, 201,   ],#[1,17,"full"],#[1, 17, "full"],
        start_on_keyframe=False,
        next2topv2=True,
        trans_box2top=False,
        base_fps=10,
        fps = [ [10],[10], [10], [10],    ],
        # repeat_times = [1],
        img_collate_param=img_collate_param_train,
        micro_frame_size=micro_frame_size,
        balance_keywords=balance_keywords,
        drop_ori_imgs=False,
    )),
    

    ((448, 800),dict(
        type = "NuplanVariableDataset",
        ann_file=["./pickle/nuplan_mini_10hz_pkl/nuplan_mini_train.pkl"],
        pipeline=pipeline2,
        dataset_root=None,
        object_classes=object_classes,
        map_classes=map_classes,
        load_interval=1,
        with_velocity=True,
        modality=input_modality,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length = [ 33, 65,  129 ],#[1,17,"full"],#[1, 17, "full"],
        start_on_keyframe=False,
        next2topv2=True,
        trans_box2top=False,
        base_fps=10,
        fps = [ [10],[10], [10],  ],
        # repeat_times = [1],
        img_collate_param=img_collate_param_train,
        micro_frame_size=micro_frame_size,
        balance_keywords=balance_keywords,
        drop_ori_imgs=False,
    )),
    
]

 
 
bucket_config = { 
    "448-800-10-1": 1,
    "448-800-10-9": 1,
    "448-800-10-17": 1,
    "448-800-10-33": 1,
    "448-800-10-65": 1,
    "448-800-10-129": 1,
    "448-800-10-201": 1,
    # "448-800-10-full": 1,
    
    "224-400-10-1": 4,
    "224-400-10-9": 2,
    "224-400-10-17": 2,
    "224-400-10-33": 2,
    "224-400-10-65": 1,
    "224-400-10-129": 1,
    "224-400-10-201": 1,
    # "224-400-10-full": 1,
    
    "112-192-10-1": 4,
    "112-192-10-9": 4,
    "112-192-10-17": 2,
    "112-192-10-33": 2,
    "112-192-10-65": 1,
    "112-192-10-129": 1,
    "112-192-10-201": 1,
    # "112-192-10-full": 1,
    
}
 
 


# Dataset settings
dataset = dict(
    type="NuplanMultiResDataset",
    cfg = dataset_cfg_list
)

validation_index = [
    # "100-448-800-10-17",
    # "200-448-800-10-17",
    # "300-448-800-10-17",
    # "700-448-800-10-17",
    # "900-448-800-10-17",
    
    
    "300-224-400-10-33",
    "300-224-400-10-65",
    "300-224-400-10-129",
    # "300-224-400-10-full",
    
    "400-224-400-10-65",
    "400-224-400-10-129",
    # "400-224-400-10-full",
    "500-224-400-10-65",
    "500-224-400-10-129",
    # "500-224-400-10-full",
    "600-224-400-10-65",
    "600-224-400-10-129",
    # "600-224-400-10-full",
    
    
    # "900-448-800-10-33",
    # "900-448-800-10-65",
    # "900-448-800-10-129",
    # "900-448-800-10-full",
]

val_dataset = dict(
    type="NuplanMultiResDataset",
    cfg = val_dataset_cfg_list
)

vae_out_channels = 16
vae=dict(
    type="VideoAutoencoderKLCogVideoX",
    from_pretrained="./ckpts/cogvideox-2b",
    subfolder="vae",
    micro_frame_size=micro_frame_size,
    micro_batch_size=1,
)

 
t_order_map = None
 
text_encoder = dict(
    type="t5",
    from_pretrained= "./ckpts/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
)


num_layers = 28
cross_attention_dim = 1152
num_attention_heads = 16
attention_head_dim = 72
global_flash_attn = True
global_layernorm = True
global_xformers = True
sp_size = 1

model = dict(
    type="DIT3DCTRL",
    num_attention_heads = num_attention_heads,
    attention_head_dim = attention_head_dim,
    in_channels = 32,
    out_channels = 16,
    num_layers = num_layers,
    norm_num_groups = 32,
    cross_attention_dim = cross_attention_dim,
    attention_bias=True,
    num_vector_embeds = None,
    patch_size = (1,2,2),
    activation_fn="gelu-approximate",
    num_embeds_ada_norm=1000,
    use_linear_projection = False,
    only_cross_attention = False,
    double_self_attention = False,
    upcast_attention = False,
    norm_type="ada_norm_single", 
    norm_elementwise_affine=False,
    norm_eps = 1e-6,
    attention_type = "default",
    caption_channels = 4096,
    use_additional_conditions = None,
    attention_mode='flash',
    downsampler = None, 
    use_rope = True,
    use_stable_fp32 = False,
    # inpaint
    vae_scale_factor_t = 4,
    class_dropout_prob = 0.1,
    model_max_length = 300,

    num_camera = 8,
    camera_control=False, 
    with_occ = False,
    with_depth = True,  
    with_seg = True,  
)

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    cog_style_trans=True,  # NOTE: trigger error with 9-frame, should change in all cases when frame > 1.
    sample_method="logit-normal",
    use_extr_loss=True
)

val = dict(
    validation_index=validation_index,
    batch_size=1,
    verbose=2,
    num_sample=1,
    save_fps=10,  # CHANGED
    seed=1024,
    scheduler = dict(
        **scheduler,
        num_sampling_steps=30,
        cfg_scale=2.5,  # base value 1, 0 is uncond
    ),
)


# Acceleration settings
num_workers = 2
num_bucket_build_workers = 4
dtype = "bf16"
plugin = "zero2-seq" if sp_size > 1 else "zero2"
batch_size = None



# Log settings
seed = 42
outputs = "outputs/uniscenev2_0403_pretrain_nuplan_all_1ref/0403_a100"
wandb = False
epochs = 1000
log_every = 100
ckpt_every = 200
sample_every = 400
load = "outputs/uniscenev2_0403_pretrain_nuplan_all_1ref/0403_a100/003-DIT3DCTRL/epoch0-global_step4200"


grad_clip = 1.0
# lr = 4e-5
lr = 6e-5
ema_decay = 0.99
adam_eps = 1e-8
weight_decay = 1e-2
warmup_steps = 0

uc_keys = ["cond_frames", "policy_trajectory", "cmd", "cond_frames_without_noise"]
diff_type = 'dit'
wo_clip = False
# round_num = 10
start_from_scratch=True
record_time=True