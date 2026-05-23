debug = False

max_epochs = 300
data_path = 'data/nuscenes/'
grad_max_norm = 10
label_mapping = './config/label_mapping/nuplan-occ.yaml'
load_from = ''

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='ReconLoss',
            weight=10.0,
            ignore_label=-100,
            use_weight=False,
            cls_weight=None,
            input_dict={
                'logits': 'logits',
                'labels': 'inputs'}),
        dict(
            type='LovaszLoss',
            weight=1.0,
            input_dict={
                'logits': 'logits',
                'labels': 'inputs'}),
        dict(
            type='VQVAEEmbedLoss',
            weight=1.0),
        ])

loss_input_convertion = dict(
    logits='logits',
    embed_loss='embed_loss'
)
_dim_ = 16
expansion = 8
base_channel = 64
n_e_ = 512
model = dict(
    type = 'VAERes2D',
    encoder_cfg=dict(
        type='Encoder2D',
        ch = base_channel, 
        out_ch = base_channel, 
        ch_mult = (1,2,4), 
        num_res_blocks = 2,
        attn_resolutions = (50,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 200, 
        z_channels = base_channel * 2, 
        double_z = False,
    ), 
    decoder_cfg=dict(
        type='Decoder2D',
        ch = base_channel, 
        out_ch = _dim_ * expansion, 
        ch_mult = (1,2,4), 
        num_res_blocks = 2,
        attn_resolutions = (50,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 200, 
        z_channels = base_channel * 2, 
        give_pre_end = False
    ),
    num_classes=18,
    expansion=expansion, 
    vqvae_cfg=dict(
        type='VectorQuantizer',
        n_e = n_e_, 
        e_dim = base_channel * 2, 
        beta = 1., 
        z_channels = base_channel * 2, 
        use_voxel=True))


multisteplr = False
multisteplr_config = dict(
    decay_t = [87 * 500],
    decay_rate = 0.1,
    warmup_t = 200,
    warmup_lr_init = 1e-6,
    t_in_epochs = False
)


optimizer = dict(optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.1))
print_freq = 10
return_len_ = 5
shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]


train_dataset_config = dict(
    data_path='/data/longhun/3D/nuplan',
    imageset='data/nuplan_mini_train_clip_infos.pkl',
    occ_dataroot='/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16',
    offset=0,
    return_len=return_len_,
    type='nuScenesSceneDatasetLidar_Nuplan',
    debug=debug)
train_loader = dict(batch_size=8, num_workers=8, shuffle=True)
train_wrapper_config = dict(phase='train', type='tpvformer_dataset_nuscenes')
unique_label = [0,1,2,3,4,5,6,7]
val_dataset_config = dict(
    data_path='/data/longhun/3D/nuplan',
    # imageset='/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val.pkl',
    imageset='data/nuplan_mini_val_clip_infos.pkl',
    occ_dataroot='/data/longhun/3D/nuplan/Nuplan-Occupancy/dataset/nuplan_occ_miniset/quan_occ/nuplan_quantized_200_200_16',
    offset=0,
    return_len=return_len_,
    type='nuScenesSceneDatasetLidar_Nuplan',
    debug=debug)
val_loader = dict(batch_size=1, num_workers=4, shuffle=False)
val_wrapper_config = dict(phase='val', type='tpvformer_dataset_nuscenes')



# train_dataset_config = dict(
#     data_path='/mnt/datasets/nuplan-all/2-0-0/dataset/nuplan-v1.1',
#     imageset='/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_train.pkl',
#     occ_dataroot='/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16',
#     offset=0,
#     return_len=5,
#     type='nuScenesSceneDatasetLidar_Nuplan',
#     debug=debug)
# train_loader = dict(batch_size=10, num_workers=12, shuffle=True)
# train_wrapper_config = dict(phase='train', type='tpvformer_dataset_nuscenes')
# unique_label = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# val_dataset_config = dict(
#     data_path='/mnt/datasets/nuplan-all/2-0-0/dataset/nuplan-v1.1',
#     imageset='/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val.pkl',
#     occ_dataroot='/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16',
#     offset=0,
#     return_len=5,
#     type='nuScenesSceneDatasetLidar_Nuplan',
#     debug=debug)
# val_loader = dict(batch_size=1, num_workers=3, shuffle=False)
# val_wrapper_config = dict(phase='val', type='tpvformer_dataset_nuscenes')

warmup_iters = 1000
work_dir = './out/VAE_nuplan_200'
# VAE Z shape: torch.Size([4, 8, 5, 50, 50]) and x shape: torch.Size([4, 5, 200, 200, 16])