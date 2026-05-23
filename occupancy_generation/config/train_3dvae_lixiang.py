debug=False  

_dim_ = 32
base_channel = 4
ch_multi_rate = 16
data_path = 'data/nuscenes/'
expansion = 4
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
loss_input_convertion = dict(
    kl_loss='kl_loss', 
    logits='logits',
)
max_epochs = 300
model_dict_square = dict(
    encoder_cfg=dict(
        type='Encoder3D_new',
        ch = 64, 
        out_ch = 8,
        ch_mult = (1,2), 
        num_res_blocks = 2,
        attn_resolutions = (100,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = 128,
        resolution = 200, 
        z_channels = 16, 
        double_z = True,
    ), 
    decoder_cfg=dict(
        type='Decoder3D_withT',
        z_channels = 16,
        ch_mult = (2,1),
        n_hiddens = 64, 
        n_res_layers =2, 
        upsample = (1,2,2),
    ),
    num_classes=10,
    expansion=4,
)

multisteplr = False  # 改为使用余弦退火
# multisteplr_config = dict(
#     decay_rate=0.9,  # 从0.8调整到0.9，衰减更温和
#     decay_t=[
#         2000,   # 延长衰减间隔
#         8000,
#         15000,
#         25000,
#         40000,
#         60000,
#         80000,
#         100000,
#         120000,
#         140000,
#         160000,
#         180000,
#     ],
#     t_in_epochs=False,
#     warmup_lr_init=1e-06,
#     warmup_t=500)  # 减少warmup步数
n_e_ = 512
num_res = 2
optimizer = dict(optimizer=dict(lr=0.00003, type='AdamW', weight_decay=0.05))  # finetune用更小的学习率
warmup_iters = 0  # finetune时跳过warmup
print_freq = 10
return_len_ = 5
shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]
train_dataset_config = dict(
    data_path='/mnt/datasets/nuplan-all/2-0-0/dataset/nuplan-v1.1',
    imageset='/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_train.pkl',
    occ_dataroot='/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16',
    offset=0,
    return_len=return_len_,
    type='nuScenesSceneDatasetLidar_Nuplan',
    debug=debug)
train_loader = dict(batch_size=8, num_workers=8, shuffle=True)
train_wrapper_config = dict(phase='train', type='tpvformer_dataset_nuscenes')
unique_label = [0,1,2,3,4,5,6,7,8
                # ,9,10,11,12,13,14,15,16
                ]
val_dataset_config = dict(
    data_path='/mnt/datasets/nuplan-all/2-0-0/dataset/nuplan-v1.1',
    # imageset='/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val.pkl',
    imageset='/mnt/volumes/ad-lmm-data-proc-bd-ga/hzhu/code/occ_gen/data/nuplan_mini_10hz_val_occ.pkl',
    occ_dataroot='/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16',
    offset=0,
    return_len=return_len_,
    type='nuScenesSceneDatasetLidar_Nuplan',
    debug=debug)
val_loader = dict(batch_size=1, num_workers=3, shuffle=False)
val_wrapper_config = dict(phase='val', type='tpvformer_dataset_nuscenes')
warmup_iters = 1000
work_dir = './out/VAE_nuplan_200'
# VAE Z shape: torch.Size([4, 8, 5, 50, 50]) and x shape: torch.Size([4, 5, 200, 200, 16])