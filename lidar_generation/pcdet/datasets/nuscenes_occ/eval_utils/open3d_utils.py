
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["render_open3d", "save_open3d_render", "render_open3d_mesh"]

def render_open3d_mesh(mesh, buffer=None, pose=np.eye(4), render=None):
  '''
  mesh can be off type o3d.gemetry or List[o3d.geometry]
  '''

  import open3d as o3d
  import open3d.visualization.rendering as rendering

  # TRICK: avoid the annoying warning messages by pass the render in
  if render is None:
    render = rendering.OffscreenRenderer(800, 800)
  else:
    render.scene.clear_geometry()
    
  mtl = rendering.MaterialRecord()
  mtl.base_color = [1, 1, 1, 1]
  mtl.point_size = 1
  mtl.shader = "defaultLit"
  mtl.sRGB_color = True

  line_mtl = rendering.MaterialRecord()
  line_mtl.shader = "unlitLine"

  render.scene.set_background([255, 255, 255, 255])

  if isinstance(mesh, list):
    for i, current_mesh in enumerate(mesh):
      # current_mesh.compute_vertex_normals()
      if type(current_mesh) == o3d.geometry.LineSet:
        render.scene.add_geometry(f'mesh{i}', current_mesh, line_mtl)
      else:   
        render.scene.add_geometry(f'mesh{i}', current_mesh, mtl)
  else:    
    # mesh.compute_triangle_normals()
    if type(current_mesh) == o3d.geometry.LineSet:
      render.scene.add_geometry(f'mesh{i}', current_mesh, line_mtl)
    else:   
      render.scene.add_geometry(f'mesh{i}', current_mesh, mtl)

  if buffer is not None:
    render.scene.add_geometry("buffer", buffer, mtl)
  
  render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))
  render.scene.scene.enable_sun_light(True)

  render.setup_camera(50.0, pose[:3,3], pose[:3,3] + [0, 0, 50], pose[:3,:3] @ np.array([0, 1, 0]))
  render.scene.view.set_post_processing(False)
  render.scene.view.set_ambient_occlusion(True)
  bev_img = render.render_to_image()
  
  render.setup_camera(50.0, pose[:3,3], pose[:3,3] + [0, 40, 20], pose[:3,:3] @ np.array([0, 0, 1]))
  render.scene.view.set_post_processing(False)
  render.scene.view.set_ambient_occlusion(True)
  pts_img = render.render_to_image()

  render.setup_camera(50.0, pose[:3,3], pose[:3,3] + [40, 0, 30], pose[:3,:3] @ np.array([0, 0, 1]))
  render.scene.view.set_post_processing(False)
  render.scene.view.set_ambient_occlusion(True)
  side_img = render.render_to_image()

  return bev_img, pts_img, side_img

def render_open3d(points, spatial_range, ultralidar=True, cond_rgb=None):

  import open3d as o3d
  import open3d.visualization.rendering as rendering

  points = points# + np.random.randn(*points.shape) * 0.001

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  if(ultralidar):
      pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(
          ((
              (points[:,2]-spatial_range[4])/(spatial_range[5] - spatial_range[4])
          )*1.4 -0.3)
          )[:,:3])
  else:
      pcd.colors = o3d.utility.Vector3dVector(plt.cm.inferno((points[:,2]+3.23)/7.0)[:,:3])

  # offscreen rendering
  render = rendering.OffscreenRenderer(2000, 1250)
  mtl = rendering.MaterialRecord()
  mtl.base_color = [1, 1, 1, 1]

  if(ultralidar):
      mtl.point_size = 2.5#1.8
  else:
      mtl.point_size = 2
  
  
  mtl.shader = "defaultLit"
  
  if(ultralidar):
      render.scene.set_background([255, 255, 255, 255])
  else:
      render.scene.set_background([0, 0, 0, 255])


  if(cond_rgb is not None):
    cond_rgb = np.flip(cond_rgb, 0)
    mesh = o3d.geometry.TriangleMesh.create_box(
        height=(100.0),
        width=(100.0),
        depth=0.01,
        create_uv_map=True,
        map_texture_to_each_face=True)
    mesh.translate(np.array([spatial_range[0], spatial_range[2], spatial_range[4]+1]))
    image = o3d.geometry.Image((cond_rgb*255).astype(np.uint8))
    mesh.textures = [image, image, image, image, image, image]
    material = rendering.MaterialRecord()
    material.shader="defaultUnlit"
    mesh.paint_uniform_color(np.array([1.0, 1.0, 1.0]))
    material.base_color=np.array([1.0, 1.0, 1.0, 1.0])
    material.sRGB_color = True
    material.albedo_img = image
    mesh.compute_triangle_normals()
    render.scene.add_geometry("floor", mesh, material)


  render.scene.add_geometry("point cloud", pcd, mtl)
  render.scene.scene.enable_sun_light(True)
  render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))

  render.setup_camera(50.0, [0, 0, 0], [0, 0, 50], [0, 1, 0])
  render.scene.view.set_post_processing(False)
  bev_img = render.render_to_image()
  
  render.setup_camera(50.0, [0, 0, 0], [0, -40, 20], [0, 0, 1])
  render.scene.view.set_post_processing(False)
  pts_img = render.render_to_image()

  render.setup_camera(50.0, [0, 0, 0], [40, 0, 30], [0, 0, 1])
  render.scene.view.set_post_processing(False)
  side_img = render.render_to_image()

  return bev_img, pts_img, side_img

def save_open3d_render(filename, img, quality=-1):
  import open3d as o3d
  o3d.io.write_image(filename, img, quality=quality)

