debug=True

grad_max_norm = 35
print_freq = 10
max_epochs = 200
warmup_iters = 200
return_len_ = 10

multisteplr = False
multisteplr_config = dict(
    decay_t = [87 * 500],
    decay_rate = 0.1,
    warmup_t = warmup_iters,
    warmup_lr_init = 1e-6,
    t_in_epochs = False
)


optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,#1e-3,
        weight_decay=0.01,
    ),
)

data_path='/mnt/datasets/nuplan-all/2-0-0/dataset/nuplan-v1.1'


train_dataset_config = dict(
    type='nuScenesSceneDatasetLidar_Nuplan_bev',
    data_path = data_path,
    return_len = return_len_, 
    offset = 0,
    occ_dataroot='/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16',
    imageset = '/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_train.pkl', 
    debug=debug
)
    
val_dataset_config = dict(
    type='nuScenesSceneDatasetLidar_Nuplan_bev',
    data_path = data_path,
    return_len = return_len_, 
    offset = 0,
    occ_dataroot='/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_200_200_16',
    imageset = '/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val.pkl', 
    debug=debug
)

train_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes_step2',
    phase='train', 
)

val_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes_step2',
    phase='val', 
)

train_loader = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)

val_loader = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)

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

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='ReconLoss',
            weight=1.0,
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
            type='KL_Loss',
            weight=1.0),
        # dict(
        #     type='VQVAEEmbedLoss',
        #     weight=1.0),
        ])

loss_input_convertion = dict(
    logits='logits',
    kl_loss ='kl_loss'
    # embed_loss='embed_loss'
)


load_from = ''



shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = './config/label_mapping/nuplan-occ.yaml'