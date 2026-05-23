import os
#os.environ['QT_DEBUG_PLUGINS'] = "1"
import pickle
import multiprocessing
import numpy as np
from pyvirtualdisplay import Display
H, W = 1080, 1920
display = Display(visible=False, size=(W, H))
display.start()
import imageio

occ_root = '/lpai/dataset/nuplan-occ/1-1-01/GT_occ_fast_val/dense_voxels_with_semantic'
#pkl_file = '/lpai/volumes/lmm-data-proc/hzhu/data/nuplan_pkls/nuplan_trainval_10hz_pkl/nuplan_10hz_trainval_part/nuplan_trainval_val_part1.pkl'
pkl_file = "/mnt/datasets/nuplan-pkl/25-03-08-1/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val.pkl"
with open(pkl_file, 'rb') as f:
    infos = pickle.load(f)
# downsample_rate = 2

classname_to_color = {  # RGB.
    0: (255, 255, 255),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle
    3: (255, 127, 80),  # Orangered bus
    4: (255, 158, 0),  # Orange car
    5: (233, 150, 70),  # Darksalmon construction
    6: (255, 61, 99),  # Red motorcycle
    7: (0, 0, 230),  # Blue pedestrian
    8: (47, 79, 79),  # Darkslategrey trafficcone
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (175, 0, 75),  # flat other
    13: (75, 0, 75),  # sidewalk
    14: (112, 180, 60),  # terrain
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
}
def custom_colormap(plt_plot, colormap=classname_to_color):
    ori_colormap = plt_plot.module_manager.scalar_lut_manager.lut.table.to_array()
    for key, value in colormap.items():
        ori_colormap[key, :3] = value
    plt_plot.module_manager.scalar_lut_manager.lut.table = ori_colormap.astype(np.uint8)[:17]


def draw(voxels, voxel_size=0.2, vis_root=None, idx=0, figure=None, mlab=None):
    if figure is None:
        figure = mlab.figure(size=(W, H), bgcolor=(1, 1, 1))
    scene = figure.scene

    camera = mlab.gcf().scene.camera
    camera.position = [0, 0, 10]
    camera.focal_point = [0, 0, 0]
    camera.view_up = [1, 0, 0]
    camera.compute_view_plane_normal()

    plt_plot = mlab.points3d(
        voxels[:, 0],
        voxels[:, 1],
        voxels[:, 2],
        voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.5 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=16,
    )
 
    scene.render()


    custom_colormap(plt_plot)

    #mlab.show()


    mlab.draw()
    vis_path = os.path.join(vis_root, '{:0>4d}.png'.format(idx))
    mlab.savefig(filename=vis_path)

def render_one_scene(scene_id, sample_tokens):
    from mayavi import mlab
    print(f'Start render scene{scene_id}')
    vis_root = "/mnt/volumes/ad-cl-rl-train-vol-ga/hzhu/hzhu/code/occ_gen/out/vis_out/test"
    os.makedirs(vis_root, exist_ok=True)
    writer = imageio.get_writer(f'{vis_root}/{scene_id}.mp4', fps=10)
    figure = mlab.figure(size=(W, H), bgcolor=(1, 1, 1))
    for i, sample_token in enumerate(sample_tokens):
        occ_path = os.path.join(occ_root, sample_token, sample_token+'.npz')
        print(occ_path)
        if not os.path.exists(occ_path):
            print(f'scene {scene_id} is not completed')
            break

        # npy
        #voxels_ = np.load(occ_path)
        
        # npz
        voxels_ = np.load(occ_path)
        voxels_ = voxels_['occ']

        # pkl
        #with open(occ_path, 'rb') as f:
        #    voxels_ = pickle.load(f)
        voxels_[:, -1] += 1
        mlab.clf()
        draw(voxels_, voxel_size=2, figure=figure, mlab=mlab, vis_root=vis_root)

        #mlab.savefig("test_output.png")

        frame = mlab.screenshot(mode='rgb', antialiased=True)
        writer.append_data(frame)
        
        print(f"{scene_id} 渲染{i}/{len(sample_tokens)}")
    mlab.close(all=True)
    writer.close()

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    #filepath = 'z_nuplan_occ.npy'
    # print(len(infos['scene_tokens']))
    # print(infos['scene_tokens'][0])

    for scene_id, sample_tokens in enumerate(infos['scene_tokens']):
        if scene_id == 1:
            print(scene_id)
            render_one_scene(scene_id, sample_tokens)

    # with multiprocessing.Pool(processes=12) as pool:
    #     args = list(zip(list(range(len(infos['scene_tokens']))), infos['scene_tokens']))
    #     pool.starmap(render_one_scene, args)


    display.stop()