debug=True

_dim_ = 32
base_channel = 4
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
            type='KL_Loss',
            weight=0.1),
        # dict(
        #     type='VQVAEEmbedLoss',
        #     weight=1.0),
        ])

loss_input_convertion = dict(kl_loss='kl_loss', logits='logits')
max_epochs = 300

expansion = 4
n_e_ = 512
ch_multi_rate = 16
# num_res  = 1
num_res  = 4
model = dict(
    type = 'VAERes3D',
    encoder_cfg=dict(
        type='Encoder3D_new',
        ch = base_channel * ch_multi_rate, 
        out_ch = base_channel*2, # useless
        ch_mult = (1,2,4), 
        num_res_blocks = num_res,
        attn_resolutions = (50,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 400, 
        z_channels = base_channel, 
        double_z = True,
    ), 
    decoder_cfg=dict(
        type='Decoder3D_withT',
        z_channels = base_channel,
        ch_mult = (4,2,1),
        n_hiddens = base_channel * ch_multi_rate, 
        n_res_layers = num_res, 
        upsample = (1,4,4),
        final_channels = _dim_ * expansion
    ),

    num_classes=18,
    expansion=expansion, 
    # vqvae_cfg=dict(
    #     type='VectorQuantizer',
    #     n_e = n_e_, 
    #     e_dim = 256, 
    #     beta = 1., 
    #     z_channels = base_channel, 
    #     use_voxel=True)
    )

multisteplr = True
multisteplr_config = dict(
    decay_rate=0.8,
    decay_t=[1000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000],
    t_in_epochs=False,
    warmup_lr_init=1e-06,
    warmup_t=1000)
optimizer = dict(optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01))
print_freq = 2
return_len_ = 5
shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

train_dataset_config = dict(
    data_path='/mnt/datasets/nuplan-all/2-0-0/dataset/nuplan-v1.1',
    imageset='/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_train.pkl',
    occ_dataroot='/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_400_400_32',
    bev_dataroot='ss',
    offset=0,
    return_len=return_len_,
    quantize_size=(400,400,32),
    type='nuScenesSceneDatasetLidar_Nuplan_pro_new_occ_bev',
    debug=debug)
train_loader = dict(batch_size=1, num_workers=0, shuffle=True)
train_wrapper_config = dict(phase='train', type='tpvformer_dataset_nuplan_pro_occ_bev')
unique_label = [0,1,2,3,4,5,6,7]

val_dataset_config = dict(
    data_path='/mnt/datasets/nuplan-all/2-0-0/dataset/nuplan-v1.1',
    imageset='/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val.pkl',
    occ_dataroot='/mnt/datasets/nuplan-occ/1-1-01/occ_quan/nuplan_quantized_400_400_32',
    bev_dataroot='ss',
    offset=0,
    return_len=return_len_,
    quantize_size=(400,400,32),
    type='nuScenesSceneDatasetLidar_Nuplan_pro_new_occ_bev',
    debug=debug)
val_loader = dict(batch_size=1, num_workers=0, shuffle=False)
val_wrapper_config = dict(phase='val', type='tpvformer_dataset_nuplan_pro_occ_bev')
warmup_iters = 1000
work_dir = './out/VAE_nuplan'
