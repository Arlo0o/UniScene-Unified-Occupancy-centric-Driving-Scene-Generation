import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d

class Occ2LiDAROpen3D(nn.Module):
    def __init__(
        self,
        model_cfg,
        **kwargs
    ):
        super().__init__()
        in_channels = model_cfg.in_channels
        unified_voxel_size = model_cfg.unified_voxel_size
        unified_voxel_shape = model_cfg.unified_voxel_shape
        self.ray_sampler_cfg = model_cfg.ray_sampler_cfg
        pc_range = model_cfg.pc_range
        if kwargs.get("fp16_enabled", False):
            self.fp16_enabled = True
        self.in_channels = in_channels
        self.pc_range = np.array(pc_range, dtype=np.float32)
        self.unified_voxel_shape = np.array(unified_voxel_shape, dtype=np.int32)
        self.unified_voxel_size = np.array(unified_voxel_size, dtype=np.float32)
        # Define vertices and faces for a single unit cube
        self.cube_vertices = torch.from_numpy(np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ])).cuda()

        self.cube_faces = torch.from_numpy(np.array([
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ])).cuda()
        self.voxel_size_cuda = torch.tensor(unified_voxel_size).cuda()
        self.pc_range_cuda = torch.tensor(pc_range).cuda()

        # generate pre-defined rays
        self.use_predefine_rays = model_cfg.get('use_predefine_rays', False)
        if self.use_predefine_rays:
            self.predefine_rays_cfg = model_cfg.predefine_rays_cfg
            azimuth_range = self.predefine_rays_cfg.azimuth_range
            azimuth_res = self.predefine_rays_cfg.azimuth_res
            elevation_range = self.predefine_rays_cfg.elevation_range
            elevation_res = self.predefine_rays_cfg.elevation_res
            elevation_beams = self.predefine_rays_cfg.elevation_beams
            azi = np.arange(azimuth_range[0], azimuth_range[1], azimuth_res)
            #ele = np.arange(elevation_range[0], elevation_range[1], elevation_res)
            ele = np.linspace(elevation_range[0], elevation_range[1], elevation_beams)

            assert len(ele) == elevation_beams, f'number of beams is not {elevation_beams}'
            
            # create meshgrid of all possible combination of azi and ele
            ae = np.vstack(np.meshgrid(azi,ele)).reshape(2,-1)

            directions = np.vstack((np.cos(np.deg2rad(ae[1,:])) * np.cos(np.deg2rad(ae[0,:])), 
                        np.cos(np.deg2rad(ae[1,:])) * np.sin(np.deg2rad(ae[0,:])),
                        np.sin(np.deg2rad(ae[1,:])))).T
            directions = directions / np.linalg.norm(directions, axis=1)[:,np.newaxis]
            origins = np.zeros_like(directions)
            self.predefine_rays = {
                'ray_o': torch.from_numpy(origins).to(torch.float32).cuda(),
                'ray_d': torch.from_numpy(directions).to(torch.float32).cuda()
            }

    def get_loss(self):
        raise NotImplementedError
    
    def generate_predicted_pc(self, batch_dict):
        batch_size = batch_dict['batch_size']
        pred_dicts = {'pc_out': [], 'gt_pts': []}
        voxel_size = self.unified_voxel_size
        pc_range = self.pc_range
        grid_size = self.unified_voxel_shape
        sensor_locs_all = batch_dict.get('sensor_loc', torch.zeros_like(batch_dict['points'][:, :4]))
        sensor_locs_all[:, -1] += 0.4
        
        o3d_cuda_device = o3d.core.Device("CUDA:0")
        o3d_cpu_device = o3d.core.Device("CPU:0")
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int64
        for bs in range(batch_size):
            batch_mask = batch_dict['occ'][:, 0].int() == bs

            fast_impl=True
            assert fast_impl
            if not fast_impl:
                assert NotImplementedError
                occ_sp = batch_dict['occ'][batch_mask][:, 1:5].int().cpu().numpy()
                occ_sp = (occ_sp[:, [2, 1, 0]]+0.5)*voxel_size[None, :]+pc_range[None, :3]
                occupied_indices = occ_sp
                # Create an Open3D mesh
                mesh = o3d.geometry.TriangleMesh()
                cube_w, cube_h, cube_d = voxel_size
                unit_cube = o3d.geometry.TriangleMesh.create_box(width=cube_w, height=cube_h, depth=cube_d)
                unit_cube.translate([-cube_w/2, -cube_h/2, -cube_d/2])  # Center the cube at the origin
                # Optionally, define the voxel size if voxels are not unit-sized

                for idx in occupied_indices:
                    # Compute the voxel's position
                    voxel_position = idx
                    
                    # Copy the unit cube and translate it to the voxel's position
                    cube = o3d.geometry.TriangleMesh(unit_cube)
                    cube.translate(voxel_position)
                    
                    # Append the cube to the voxel mesh
                    mesh += cube

                # # Optionally compute vertex normals for better visualization
                mesh.compute_vertex_normals()
                # Convert the mesh to Open3D's tensor format
                mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

                # Create a raycasting scene
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(mesh_tensor)

                # Generate rays
                pts_all = batch_dict['points']
                gt_batch_mask = pts_all[:, 0].int()==bs
                pts = pts_all[gt_batch_mask][:, 1:4]
                lidar_rays = self.sample_lidar_rays([pts])
                gt_pts = lidar_rays[0]['depth'] * lidar_rays[0]['ray_d']
                directions = lidar_rays[0]['ray_d'].cpu().numpy()
                origins = lidar_rays[0]['ray_o'].cpu().numpy()

                # Prepare rays in Open3D tensor format
                rays = o3d.core.Tensor(np.hstack((origins, directions)), dtype=o3d.core.Dtype.Float32)

                # Perform ray casting
                results = scene.cast_rays(rays)

                # Get hit distances (t_hit) and filter valid hits
                t_hits = results['t_hit'].numpy()
                hit_mask = np.isfinite(t_hits)
                t_hits = t_hits[hit_mask]

                # Get corresponding origins and directions for valid hits
                origins_hit = origins[hit_mask]
                directions_hit = directions[hit_mask]

                # Compute intersection points
                points_hit = origins_hit + directions_hit * t_hits[:, np.newaxis]

                # Visualize the point cloud
                #pcd = o3d.geometry.PointCloud()
                #pcd.points = o3d.utility.Vector3dVector(points)
                #pcd.paint_uniform_color([1, 0.706, 0])  # Example color
                #o3d.visualization.draw_geometries([mesh, pcd])

                pred_dicts['pc_out'].append(torch.from_numpy(points_hit).to(pts.device))
                pred_dicts['gt_pts'].append(gt_pts)
            else:
                occ_sp = batch_dict['occ'][batch_mask][:, 1:5]
                occ_sp = (occ_sp[:, [2, 1, 0]]+0.5)*self.voxel_size_cuda[None, :]+self.pc_range_cuda[None, :3]
                occupied_indices = occ_sp
                # Scale and translate the cube vertices for all occupied voxels
                occupied_positions = occupied_indices
                scaled_vertices = self.cube_vertices * self.unified_voxel_size[0] + occupied_positions[:, None, :]

                # Create a mesh from the scaled vertices and faces
                num_cubes = len(occupied_positions)
                all_vertices = scaled_vertices.reshape(-1, 3)
                face_offsets = torch.arange(num_cubes, device=occ_sp.device)[:, None, None] * 8
                all_faces = self.cube_faces[None, :, :] + face_offsets
                all_faces = all_faces.reshape(-1, 3)

                # Create the final mesh
                mesh = o3d.t.geometry.TriangleMesh(o3d_cpu_device)
                mesh.vertex.positions = o3d.core.Tensor(all_vertices.cpu().numpy(), dtype_f, o3d_cpu_device)
                mesh.triangle.indices = o3d.core.Tensor(all_faces.cpu().numpy(), dtype_i, o3d_cpu_device)
                # mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
                # mesh.triangles = o3d.utility.Vector3iVector(all_faces)

                # Optionally compute vertex normals
                mesh.compute_vertex_normals()

                # Create a raycasting scene
                # This class supports only the CPU device :(
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(mesh)

                # Generate rays
                pts_all = batch_dict['points']
                gt_batch_mask = pts_all[:, 0].int()==bs
                pts = pts_all[gt_batch_mask][:, 1:]
                sensor_locs = sensor_locs_all[gt_batch_mask][:, 1:]
                lidar_rays = self.sample_lidar_rays([pts], sensor_locs=[sensor_locs])
                if not self.use_predefine_rays:
                    gt_pts = lidar_rays[0]['depth'] * lidar_rays[0]['ray_d'] + lidar_rays[0]['ray_o']
                else:
                    dis = torch.norm(pts[:, :3], p=2, dim=-1)
                    dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                        dis < self.ray_sampler_cfg.get("far_radius", 100.0)
                    )
                    gt_pts = pts[dis_mask]
                directions = lidar_rays[0]['ray_d'].cpu().numpy()
                origins = lidar_rays[0]['ray_o'].cpu().numpy()

                # Prepare rays in Open3D tensor format
                rays = o3d.core.Tensor(np.hstack((origins, directions)), dtype=o3d.core.Dtype.Float32, device=o3d_cpu_device)

                # Perform ray casting
                results = scene.cast_rays(rays)

                # Get hit distances (t_hit) and filter valid hits
                t_hits = results['t_hit'].numpy()
                hit_mask = np.isfinite(t_hits)
                t_hits = t_hits[hit_mask]

                # Get corresponding origins and directions for valid hits
                origins_hit = origins[hit_mask]
                directions_hit = directions[hit_mask]

                # Compute intersection points
                points_hit = origins_hit + directions_hit * t_hits[:, None]

                # Visualize the point cloud
                #pcd = o3d.geometry.PointCloud()
                #pcd.points = o3d.utility.Vector3dVector(points)
                #pcd.paint_uniform_color([1, 0.706, 0])  # Example color
                #o3d.visualization.draw_geometries([mesh, pcd])

                lidar_idx = lidar_rays[0]['lidar_idx']
                gt_pts = torch.cat([gt_pts, lidar_idx.unsqueeze(-1)], dim=-1)
                pred_dicts['pc_out'].append(torch.from_numpy(points_hit).cuda())
                pred_dicts['gt_pts'].append(gt_pts[hit_mask])

        return pred_dicts
    
    def sample_lidar_rays(self, pts, sensor_locs=None, test=False):
        """Get lidar ray
        Returns:
            lidar_ret: list of dict, each dict contains:
                ray_o: (num_rays, 3)
                ray_d: (num_rays, 3)
                depth: (num_rays, 1)
                scaled_points: (num_rays, 3)
        """
        lidar_ret = []

        if self.use_predefine_rays and (not self.training):
            for i in range(len(pts)):
                rays = copy.deepcopy(self.predefine_rays)
                rays.update({'intersection_mask': None})
                lidar_ret.append(rays)
            return lidar_ret

        for i in range(len(pts)):
            lidar_pc = pts[i]
            dis = torch.norm(lidar_pc[:, :3], p=2, dim=-1)
            dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                dis < self.ray_sampler_cfg.get("far_radius", 100.0)
            )
            lidar_pc = lidar_pc[dis_mask]
            lidar_points = lidar_pc[:, :3]
            if sensor_locs is None:
                lidar_origins = torch.zeros_like(lidar_points)
            else:
                lidar_origins = sensor_locs[i][dis_mask]
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            lidar_ret.append(
                {
                    "ray_o": lidar_origins,
                    "ray_d": lidar_directions,
                    "depth": lidar_ranges if not test else None,
                    "scaled_points": lidar_points,
                    "scale_factor": 1.0,
                    "lidar_idx": lidar_pc[:, -1].int()
                }
            )
        return lidar_ret
        
    def forward(self, batch_dict):
        if self.training:
            raise NotImplementedError
        else:
            return batch_dict