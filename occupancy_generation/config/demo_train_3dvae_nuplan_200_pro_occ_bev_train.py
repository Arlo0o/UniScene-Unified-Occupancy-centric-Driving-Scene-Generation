debug=False  

_dim_ = 16
base_channel = 4
ch_multi_rate = 16
data_path = 'data/nuscenes/'
expansion = 8
grad_max_norm = 1
label_mapping = './config/label_mapping/nuplan-occ.yaml'
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
max_epochs = 500
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
        type='Encoder3D_new',
        ch=64,
        out_ch=8,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(50, ),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=128,
        z_channels=4,
        double_z=True,
        resolution=200,
        ),
    expansion=8,
    num_classes=18,
    type='VAERes3D')
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
optimizer = dict(optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.1))
print_freq = 1
return_len_ = 5
shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]
train_dataset_config = dict(
    data_path='/data/longhun/3D/nuplan',
    imageset='/data/longhun/3D/nuplan/Nuplan-Occupancy/pickle/mini/nuplan_mini_10hz_train.pkl',
    occ_dataroot='/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16',
    bev_dataroot="/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200",
    offset=0,
    return_len=return_len_,
    type='nuScenesSceneDatasetLidar_Nuplan_pro_new_occ_bev',
    debug=debug)
train_loader = dict(batch_size=2, num_workers=2, shuffle=True)
# train_loader = dict(batch_size=1, num_workers=1, shuffle=True)
train_wrapper_config = dict(phase='train', type='tpvformer_dataset_nuplan_pro_occ_bev')
unique_label = [0,1,2,3,4,5,6,7]


warmup_iters = 1000
work_dir = './out/VAE_nuplan_200'
# VAE Z shape: torch.Size([4, 8, 5, 50, 50]) and x shape: torch.Size([4, 5, 200, 200, 16])

val_dataset_config = dict(
    data_path='/data/longhun/3D/nuplan',
    imageset='data/nuplan_mini_val_clip_infos.pkl',
    occ_dataroot='/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16',
    bev_dataroot="/data/zhuhu/3DVision_datasets/Occ/nuplan/bev/sample_200",
    offset=0,
    return_len=return_len_,
    type='nuScenesSceneDatasetLidar_Nuplan_pro_new_occ_bev',
    debug=debug)
val_loader = dict(batch_size=1, num_workers=1, shuffle=False)
val_wrapper_config = dict(phase='val', type='tpvformer_dataset_nuplan_pro_occ_bev')