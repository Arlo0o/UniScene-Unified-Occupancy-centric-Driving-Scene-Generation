_dim_ = 32
base_channel = 4
ch_multi_rate = 16
expansion = 4
gpu_ids = range(0, 8)
grad_max_norm = 35
label_mapping = '/code/code/Diff_occ/occ_gen/occ_gen/nus_pros/nuscene.yaml'
load_from = ''

loss = dict(
    loss_cfgs=[
        dict(
            cls_weight=None,
            ignore_label=-100,
            input_dict=dict(labels='inputs', logits='logits'),
            type='ReconLoss',
            use_weight=False,
            weight=10.0),
        dict(
            input_dict=dict(labels='inputs', logits='logits'),
            type='LovaszLoss',
            weight=1.0),
        dict(type='KL_Loss', weight=0.1),
    ],
    type='MultiLoss')

loss_input_convertion = dict(kl_loss='kl_loss', logits='logits')

max_epochs = 100

model = dict(
    decoder_cfg=dict(
        ch_mult=(
            4,
            2,
            1,
        ),
        final_channels=128,
        n_hiddens=64,
        n_res_layers=1,
        type='Decoder3D_withT',
        upsample=(
            1,
            4,
            4,
        ),
        z_channels=4),
    encoder_cfg=dict(
        attn_resolutions=(50, ),
        ch=64,
        ch_mult=(
            1,
            2,
            4,
        ),
        double_z=True,
        dropout=0.0,
        in_channels=128,
        num_res_blocks=1,
        out_ch=8,
        resamp_with_conv=True,
        resolution=400,
        type='Encoder2D_new2',
        z_channels=4),
    expansion=4,
    num_classes=17,
    type='VAERes2D_DwT')

multisteplr = True
multisteplr_config = dict(
    decay_rate=0.8,
    decay_t=[
        1000,
        10000,
        20000,
        30000,
        40000,
        50000,
        60000,
        70000,
        80000,
        90000,
        100000,
        110000,
    ],
    t_in_epochs=False,
    warmup_lr_init=1e-06,
    warmup_t=200)
n_e_ = 512
num_res = 1
nusc_ori_root = '/code/code/Diff_occ/occ_gen/occ_gen/data/nuscenes'
occ_base_path = '/data/longhun/3D/nuscenes/data/pyramid_occ/nuscene_quantized_400_400_32/quantized'
optimizer = dict(optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01))
print_freq = 10
return_len_ = 8
shapes = [
    [
        200,
        200,
    ],
    [
        100,
        100,
    ],
    [
        50,
        50,
    ],
    [
        25,
        25,
    ],
]
train_dataset_config = dict(
    imageset=
    '/code/code/Diff_occ/occ_gen/occ_gen/data/nuscenes/nuscenes_interp_12Hz_infos_train.pkl',
    nusc_dataroot=nusc_ori_root,
    occ_base_path=occ_base_path,
    offset=0,
    quantize_size=(
        400,
        400,
        32,
    ),
    return_len=8,
    type='nuScenesSceneDatasetLidar_HR')
train_loader = dict(batch_size=1, num_workers=2, shuffle=False)
train_wrapper_config = dict(
    phase='train', type='tpvformer_dataset_nuscenes_HR_woBev')
unique_label = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
]
val_dataset_config = dict(
    imageset=
    '/code/code/Diff_occ/occ_gen/occ_gen/data/nuscenes/nuscenes_interp_12Hz_infos_train.pkl',
    nusc_dataroot=nusc_ori_root,
    occ_base_path=occ_base_path,
    offset=0,
    quantize_size=(
        400,
        400,
        32,
    ),
    return_len=8,
    type='nuScenesSceneDatasetLidar_HR')
val_loader = dict(batch_size=1, num_workers=2, shuffle=False)
val_wrapper_config = dict(
    phase='val', type='tpvformer_dataset_nuscenes_HR_woBev')
warmup_iters = 200
work_dir = '/code/code/Diff_occ/occ_gen/occ_gen/out_12hz/eval_vae_c16r1_400'
