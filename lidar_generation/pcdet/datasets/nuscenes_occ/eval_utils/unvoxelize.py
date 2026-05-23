
import torch
import numpy as np



def unvoxelize(voxels, spatial_range, voxel_size):
    occupancy = torch.where(voxels>.5, 1.0, 0.0)

    if(len(voxels.shape) > 3):
        occupancy = occupancy[0]
        
    mesh_x = torch.arange(spatial_range[0], spatial_range[1], voxel_size[0]).cuda()
    mesh_y = torch.arange(spatial_range[2], spatial_range[3], voxel_size[1]).cuda()
    mesh_z = torch.arange(spatial_range[4], spatial_range[5], voxel_size[2]).cuda()

    mesh = torch.stack(torch.meshgrid(mesh_z, mesh_x, mesh_y)).float()

    indice = occupancy.nonzero(as_tuple=True)
    x, y, z = indice[0], indice[1], indice[2]
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    flat_indice = np.ravel_multi_index([x,y,z], (occupancy.shape[0], occupancy.shape[1], occupancy.shape[2]))

    values = torch.reshape(mesh, (3, -1,))[:,flat_indice]

    values = values[[2, 1, 0]]

    return values.T



def unvoxelize_field(occupancy, field, spatial_range, voxel_size): 
    occupied = torch.where(occupancy>.5, 1.0, 0.0)

    if(len(occupancy.shape) > 3):
        occupied = occupied[0]

    if(len(field.shape) ==3):
        field = field.unsqueeze(0)
        
    mesh_x = torch.arange(spatial_range[0], spatial_range[1], voxel_size[0])
    mesh_y = torch.arange(spatial_range[2], spatial_range[3], voxel_size[1])
    mesh_z = torch.arange(spatial_range[4], spatial_range[5], voxel_size[2])

    mesh = torch.stack(torch.meshgrid(mesh_z, mesh_x, mesh_y)).float()
    mesh = torch.cat([mesh, field.detach().cpu()], 0)

    indice = occupied.nonzero(as_tuple=True)
    x, y, z = indice[0], indice[1], indice[2]
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    flat_indice = np.ravel_multi_index([x,y,z], (occupied.shape[0], occupied.shape[1], occupied.shape[2]))

    values = torch.reshape(mesh, (4, -1,))[:,flat_indice]

    values = values[[2, 1, 0, 3]]

    return values.T