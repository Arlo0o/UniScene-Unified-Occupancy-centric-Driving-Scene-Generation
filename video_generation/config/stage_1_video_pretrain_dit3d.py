from copy import deepcopy
grad_checkpoint = True
num_frames = None
micro_frame_size = None#8
bbox_mode = 'all-xyz'
template = "A driving scene video at {location}. {description}."
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

balance_keywords = None#["night", "rain", "none"]
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
    "timeofday",
    "location",
    "description",
    "filename",
    "token",
] # hold by one DataContainer
view_order = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles",
        to_float32=True
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
        safe=False
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
            "lidar",
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
    final_dim=[424, 800],
    resize_lim=[0.5, 0.5],
    bot_pct_lim = [0.0, 0.0],
    rot_lim='null',
    rand_flip=False,
    is_train=False
)



dataset_cfg_list = [
    ((224, 400),dict(
        type = "NuScenesVariableDataset",
        ann_file="/code/world_model/WorldModel-uniscenev2_sim/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl",
        pipeline=pipeline,
        dataset_root="/code/data/occupancy/OpenOccupancy-main/data/nuscenes/",
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
        force_all_boxes=True,
        video_length = [9, 17, 33 ], #[1,17,"full"],#[1, 17, "full"],
        start_on_keyframe=False,
        next2topv2=True,
        trans_box2top=False,
        base_fps=12,
        fps = [[12,], [12], [12]],#[[12,], [12], [12], [12]],#[[120,], [12,], [12,],],
        # repeat_times = [1],#[1, 1, 40],
        img_collate_param=img_collate_param_train,
        micro_frame_size=micro_frame_size,
        balance_keywords=balance_keywords,
        drop_ori_imgs=False,
    )),

]

val_dataset_cfg_list = [
    ((224, 400),dict(
        type = "NuScenesVariableDataset",
        ann_file="/code/world_model/WorldModel-uniscenev2_sim/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl",
        pipeline=pipeline,
        dataset_root="/code/data/occupancy/OpenOccupancy-main/data/nuscenes/",
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
        force_all_boxes=True,
        video_length = [33,],#[1,17,"full"],#[1, 17, "full"],
        start_on_keyframe=False,
        next2topv2=True,
        trans_box2top=False,
        base_fps=12,
        fps = [[12,]],
        # repeat_times = [1],
        img_collate_param=img_collate_param_train,
        micro_frame_size=micro_frame_size,
        balance_keywords=balance_keywords,
        drop_ori_imgs=False,
    )),
   
]

bucket_config = { 
    "224-400-12-9": 2,
    "224-400-12-17": 2,
    "224-400-12-33": 1,
    "224-400-12-65": 1,
}


# bucket_config = { 
#     "224-400-120-1": 10,
#     "224-400-12-17": 4,
#     "224-400-12-33": 1,
#     "224-400-12-full": 1,  # 17-20s/it, variable length, must be 1

#     "424-800-120-1": 10,
#     "424-800-12-17": 2,  # 6: 32-34s/it, 8: 47s/it
#     "424-800-12-33": 1,  # 3: 30s/it, 4: 37-40s/it
#     "424-800-12-65": 2,  # 1: 16-18s/it, 2: 34-35s/it
#     # "424-800-12-65": 1,
#     "424-800-12-129": 1,  # 1: 34-38s/it
#     # "424-800-12-129": -1,

#     "848-1600-120-1": 10,  # 32s/it
#     "848-1600-12-9": 3,  # 3: 38-42s/it, 4: 50s/it
#     "848-1600-12-17": 2,  # 1: 20s/it, 2: 39-41s/it
#     "848-1600-12-33": 1,  # 36-40s/it
# }



# Dataset settings
dataset = dict(
    type="NuScenesMultiResDataset",
    cfg = dataset_cfg_list
)

validation_index = [
    #  "1828-848-1600-12-17",
    #  "5543-848-1600-12-17",
    #  "6720-848-1600-12-17",
    # "14449-848-1600-12-17",

    #  "5538-848-1600-12-33",
    # "14631-848-1600-12-33",
    #  "6720-848-1600-12-33",
    # "14449-848-1600-12-33",
    #  "3649-848-1600-12-33",  # know

    #  "912-424-800-12-129",
    # "1680-424-800-12-129",
    # "3657-424-800-12-129",

    #  "24-224-400-12-full",
    # "145-224-400-12-full",
    # "105-224-400-12-full",

    #  "24-224-400-12-17",
    # "145-224-400-12-17",
    # "105-224-400-12-17",
    #  "912-224-400-12-17",
    # "1680-224-400-12-17",
    # "3657-224-400-12-17",
    #  "5543-224-400-12-17",
    "14449-224-400-12-33",
    "3649-224-400-12-33",

    # "8726-848-1600-120-1",

    # "5543-424-800-12-33",  # know
    # "5543-848-1600-12-33",  # know

]

val_dataset = dict(
    type="NuScenesMultiResDataset",
    cfg = val_dataset_cfg_list
)

vae_out_channels = 16
vae=dict(
    type="VideoAutoencoderKLCogVideoX",
    from_pretrained="/code/world_model/WorldModel-uniscenev2_sim/ckpts/cogvideox-2b",
    subfolder="vae",
    micro_frame_size=micro_frame_size,
    micro_batch_size=1,
)

mv_order_map = {
    0: [5, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 0],
}
t_order_map = None

text_encoder = dict(
    type="t5",
    from_pretrained= "/code/world_model/WorldModel-uniscenev2_sim/ckpts/t5-v1_1-xxl",
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
    type="DIT3D",
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
    # bbox_embedder_param=dict(
    #     n_classes=10,
    #     class_token_dim=cross_attention_dim,
    #     trainable_class_token=False,
    #     embedder_num_freq=4,
    #     proj_dims=[cross_attention_dim, 512, 512, cross_attention_dim],
    #     mode = bbox_mode,
    #     minmax_normalize=False,
    #     use_text_encoder_init=True, 
    #     after_proj=True,
    #     sample_id=True,  # CHANGED
    #     # new
    #     num_heads=8,
    #     mlp_ratio=4.0,
    #     qk_norm=True,
    #     enable_flash_attn=False,
    #     enable_xformers=True,
    #     enable_layernorm_kernel=True,
    #     use_scale_shift_table=True,
    #     time_downsample_factor=4.5,
    # ),
    num_camera = 6,
    camera_control=False
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
    save_fps=12,  # CHANGED
    seed=1024,
    scheduler = dict(
        **scheduler,
        num_sampling_steps=30,
        cfg_scale=2.5,  # base value 1, 0 is uncond
    ),
)

# Mask settings
# mask_ratios = {
#     "random": 0.01,
#     "intepolate": 0.002,
#     # "quarter_random": 0.002,
#     "quarter_head": 0.002,
#     "quarter_tail": 0.002,
#     "quarter_head_tail": 0.002,
#     # "image_random": 0.0,
#     "image_head": 0.22,
#     "image_tail": 0.005,
#     "image_head_tail": 0.005,
# }

# mask_ratios = {
#     "quarter_random": 0.1,
# }

# Acceleration settings
num_workers = 1
num_bucket_build_workers = 1
dtype = "bf16"
plugin = "zero2-seq" if sp_size > 1 else "zero2"
batch_size = None
drop_cond_ratio = 0.1


# Log settings
seed = 42
outputs = "./outputs/uniscenev2_v1_control_check_0107_4_gpu_pretrain/0122"
wandb = False
epochs = 1000
log_every = 1
ckpt_every = 1000
sample_every = 1000
load = "outputs/uniscenev2_v1_control_check_0107_4_gpu_pretrain/0122/000-DIT3D/epoch2-global_step170000"

# save_dir = "./outputs/uniscenev2_v1_control_check_0107_4_gpu"
grad_clip = 1.0
# lr = 4e-5
lr = 1e-5
ema_decay = 0.99
adam_eps = 1e-3
weight_decay = 1e-2
warmup_steps = 0

uc_keys = ["cond_frames", "policy_trajectory", "cmd", "cond_frames_without_noise"]
diff_type = 'dit'
wo_clip = False
# round_num = 10
start_from_scratch=False
record_time=True