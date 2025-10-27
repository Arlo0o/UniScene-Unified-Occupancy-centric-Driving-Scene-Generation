import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .render_utils import apply_depth_colormap, apply_semantic_colormap

def render(extrinsics, intrinsics, image_shape,
           pts_xyz, pts_rgb, feat, rotations, scales, opacity, bg_color):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    height, width = image_shape

    # Set up rasterization configuration
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)
    tan_fov_x = math.tan(FovX * 0.5)
    tan_fov_y = math.tan(FovY * 0.5)

    extrinsics = torch.inverse(extrinsics) # w2c

    # projection_matrix = get_projection_matrix(near=0.1, far=200.0, fov_x=FovX, fov_y=FovY).transpose(0, 1).cuda()
    projection_matrix = get_projection_matrix_c(fx, fy, cx, cy, width, height, 0.1, 200.0).transpose(0, 1).cuda()
    world_view_transform = extrinsics.transpose(0, 1).cuda()
    full_projection = world_view_transform.float() @ projection_matrix

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_projection,
        sh_degree=3,
        campos=world_view_transform.inverse()[3, :3],
        prefiltered=False,
        debug=False,
        include_feature=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_feat, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pts_rgb,
        language_feature_precomp=feat,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    none_mask = rendered_alpha[0] < 0.10
    none_label = torch.zeros(20).cuda()
    none_label[0] = 1
    rendered_feat[:, none_mask] = none_label[:, None]
    rendered_depth[:, none_mask] = 51.2

    return {
        'render_color': rendered_image,
        'radii': radii,
        'render_depth': rendered_depth,
        'render_alpha': rendered_alpha,
        'render_feat': rendered_feat,
    }


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def get_projection_matrix(near, far, fov_x, fov_y):
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = math.tan(0.5 * fov_x)
    tan_fov_y = math.tan(0.5 * fov_y)

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    result = torch.zeros((4, 4), dtype=torch.float32)

    result[0, 0] = 2 * near / (right - left)
    result[1, 1] = 2 * near / (top - bottom)
    result[0, 2] = (right + left) / (right - left)
    result[1, 2] = (top + bottom) / (top - bottom)
    result[3, 2] = 1
    result[2, 2] = far / (far - near)
    result[2, 3] = -(far * near) / (far - near)

    return result

def get_projection_matrix_c(fx, fy, cx, cy, W, H, znear, zfar):
    top = cy * znear / fy
    bottom = -(H - cy) * znear / fy

    right = cx * znear / fx
    left = -(W - cx) * znear / fx

    P = torch.zeros(4, 4)

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P