from pyvirtualdisplay import Display
display = Display(visible=False, size=(2560, 1440))
display.start()

from mayavi import mlab
import mayavi
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import time, argparse, os.path as osp, os
import torch, numpy as np

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmengine.registry import MODELS

import warnings
warnings.filterwarnings("ignore")

def pass_print(*args, **kwargs):
    pass

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def draw(
    voxels,          # semantic occupancy predictions
    pred_pts,        # lidarseg predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dir=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
):
    w, h, z = voxels.shape

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 17)
    ]
    print(len(fov_voxels))
    
    # import pdb; pdb.set_trace()
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # mayavi.mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    voxel_size = sum(voxel_size) / 3
    plt_plot_fov = mlab.points3d(
        # fov_voxels[:, 1],
        # fov_voxels[:, 0],
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        scale_factor=1.0 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=16, # 16
    )

    colors = np.array(
        [
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.uint8)
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    mlab.savefig(os.path.join(save_dir, f'vis_{timestamp}.png'))
    mlab.close()

def main(args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # load config
    cfg = Config.fromfile(args.vae_config)
    cfg.work_dir = args.work_dir
    
    os.makedirs(args.work_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{cfg.get("data_type", "gts")}_visualize_autoreg_{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    my_model = MODELS.build(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    my_model = my_model.cuda()
    raw_model = my_model
    logger.info('done ddp model')
    from dataset import get_dataloader
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False)
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    epoch = 'last'
    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        epoch = ckpt['epoch']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    # eval
    my_model.eval()
    os.environ['eval'] = 'true'
    recon_dir = os.path.join(args.work_dir, args.dir_name+f'{cfg.get("data_type", "gts")}_autoreg', str(epoch))
    os.makedirs(recon_dir, exist_ok=True)
    dataset = cfg.val_dataset_config['type']
    recon_dir = os.path.join(recon_dir, dataset)
    start_frame = 48
    with torch.no_grad():
        for i_iter_val, (input_occs, target_occs, metas) in enumerate(val_dataset_loader):
            if i_iter_val not in args.scene_idx:
                continue
            if i_iter_val > max(args.scene_idx):
                break
            '''
            if i_iter_val < start_frame:
                continue'''
            input_occs = input_occs.cuda()
            if args.onlyvqvae==True:
                result = my_model(x=input_occs, metas=metas)
                # input_occs = input_occs[0]
            else:    
                result = my_model.forward_autoreg_with_pose(
                        x=input_occs, metas=metas, 
                        start_frame=cfg.get('start_frame', 0),
                        mid_frame=cfg.get('mid_frame', 5),
                        end_frame=cfg.get('end_frame', 11))
                input_occs = result['input_occs']
            # print(input_occs.shape)

            logits = result['logits']
            n_frames = logits.shape[1]
            print(n_frames,input_occs.shape[1])
            dst_dir = os.path.join(recon_dir, str(i_iter_val))
            input_dir = os.path.join(recon_dir, f'{i_iter_val}_input')
            
            os.makedirs(dst_dir, exist_ok=True)
            os.makedirs(input_dir, exist_ok=True)
            # assert n_frames < input_occs.shape[1]
            # for frame in range(n_frames+1):
            for frame in range(n_frames):
                input_occ = input_occs[:, frame, ...].squeeze().cpu().numpy()
                draw(input_occ, 
                    None, # predict_pts,
                    [-40, -40, -1], 
                    [0.4] * 3, 
                    None, #  grid.squeeze(0).cpu().numpy(), 
                    None,#  pt_label.squeeze(-1),
                    input_dir,#recon_dir,
                    None, # img_metas[0]['cam_positions'],
                    None, # img_metas[0]['focal_positions'],
                    timestamp=str(i_iter_val) + '_' + str(frame),
                    mode=0,
                    sem=False)
                if frame == n_frames:
                    continue
                logit = logits[:, frame, ...]
                print(pred.shape)
                pred = logit.argmax(dim=-1).squeeze().cpu().numpy() # 1, 1, 200, 200, 16
                draw(pred, 
                    None, # predict_pts,
                    [-40, -40, -1], 
                    [0.4] * 3, 
                    None, #  grid.squeeze(0).cpu().numpy(), 
                    None,#  pt_label.squeeze(-1),
                    dst_dir,#recon_dir,
                    None, # img_metas[0]['cam_positions'],
                    None, # img_metas[0]['focal_positions'],
                    timestamp=str(i_iter_val) + '_' + str(frame),
                    mode=0,
                    sem=False)
            logger.info('[EVAL] Iter %5d / %5d'%(i_iter_val, len(val_dataset_loader)))
            logger.info(f'gt_poses_{result["gt_poses_"]}')
            logger.info(f'poses_{result["poses_"]}')
            

if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vae-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--dir-name', type=str, default='vis')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--frame-idx', nargs='+', type=int, default=[0, 10])
    parser.add_argument('--scene-idx', nargs='+', type=int, default=[6,7,16,18,19,87,89,96,101])
    parser.add_argument('--onlyvqvae', action='store_true') # add
    args = parser.parse_args()
    
    ngpus = 1
    args.gpus = ngpus
    print(args)
    # os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

    main(args)
