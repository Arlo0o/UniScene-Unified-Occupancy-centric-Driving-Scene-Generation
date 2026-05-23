_dim_ = 16
base_channel = 4
ch_multi_rate = 16
data_path = 'data/nuscenes/'
expansion = 8
grad_max_norm = 10
label_mapping = './config/label_mapping/nuscenes-occ.yaml'
load_from = ''
loss = dict(
    loss_cfgs=[
        dict(
            cls_weight=None,
            ignore_label=-100,
            input_dict=dict(labels='inputs', logits='logits'),
            type='ReconLoss',
            use_weight=False,
            weight=1.0),
        dict(
            input_dict=dict(labels='inputs', logits='logits'),
            type='LovaszLoss',
            weight=1.0),
        dict(type='KL_Loss', weight=1.0),
    ],
    type='MultiLoss')
loss_input_convertion = dict(kl_loss='kl_loss', logits='logits')
max_epochs = 300
model = dict(
    decoder_cfg=dict(
        ch_mult=(
            4,
            2,
            1,
        ),
        n_hiddens=64,
        n_res_layers=2,
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
        num_res_blocks=2,
        out_ch=8,
        resamp_with_conv=True,
        resolution=200,
        type='Encoder2D_new2',
        z_channels=4),
    expansion=8,
    num_classes=18,
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
    warmup_t=1000)
n_e_ = 512
num_res = 2
optimizer = dict(optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.1))
print_freq = 10
return_len_ = 5
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
    data_path='/lpai/volumes/lmm-data-proc/hzhu/code/occ_gen/data/dataset/openscene-v1.0',
    imageset='/lpai/volumes/lmm-data-proc/hzhu/data/nuplan_pkls/nuplan_pkls/openscene_pkl/openscene_mini_val_v1_scene_token.pkl',
    nusc_dataroot='/lpai/volumes/lmm-data-proc/hzhu/code/occ_gen/data/dataset/openscene-v1.0',
    offset=0,
    return_len=5,
    type='nuScenesSceneDatasetLidar_OpenScene')
train_loader = dict(batch_size=4, num_workers=2, shuffle=True)
train_wrapper_config = dict(phase='train', type='tpvformer_dataset_nuscenes')
unique_label = [
    0,
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
    data_path='/lpai/volumes/lmm-data-proc/hzhu/code/occ_gen/data/dataset/openscene-v1.0',
    imageset='/lpai/volumes/lmm-data-proc/hzhu/data/nuplan_pkls/nuplan_pkls/openscene_pkl/openscene_mini_val_v1_scene_token.pkl',
    nusc_dataroot='/lpai/volumes/lmm-data-proc/hzhu/code/occ_gen/data/dataset/openscene-v1.0',
    offset=0,
    return_len=5,
    type='nuScenesSceneDatasetLidar_OpenScene')
val_loader = dict(batch_size=1, num_workers=3, shuffle=False)
val_wrapper_config = dict(phase='val', type='tpvformer_dataset_nuscenes')
warmup_iters = 1000
work_dir = './out/VAE_OpenScene'
