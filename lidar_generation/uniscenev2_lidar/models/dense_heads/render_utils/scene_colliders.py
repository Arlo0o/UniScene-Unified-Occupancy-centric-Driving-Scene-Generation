import torch
import torch.nn as nn


class SceneCollider(nn.Module):
    """Module for setting near and far values for rays."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def set_nears_and_fars(self, ray_bundle):
        """To be implemented."""
        raise NotImplementedError

    def forward(self, ray_bundle):
        """Sets the nears and fars if they are not set already."""
        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            return ray_bundle
        return self.set_nears_and_fars(ray_bundle)


class AABBBoxCollider(SceneCollider):
    """Module for colliding rays with the scene box to compute near and far values.

    Args:
        scene_box: scene box to apply to dataset
    """

    def __init__(self, scene_box, scale_factor, near_plane, **kwargs):
        super().__init__(**kwargs)
        self.scene_box = scene_box * scale_factor
        self.near_plane = near_plane * scale_factor

    def _intersect_with_aabb(
        self, rays_o, rays_d, aabb
    ):
        """Returns collection of valid rays within a specified near/far bounding box along with a mask
        specifying which rays are valid

        Args:
            rays_o: (num_rays, 3) ray origins, scaled
            rays_d: (num_rays, 3) ray directions
            aabb: (6, ) This is [min point (x,y,z), max point (x,y,z)], scaled
        """
        # avoid divide by zero
        dir_fraction = 1.0 / (rays_d + 1e-6)

        # x
        t1 = (aabb[0] - rays_o[..., 0:1]) * dir_fraction[..., 0:1]
        t2 = (aabb[3] - rays_o[..., 0:1]) * dir_fraction[..., 0:1]
        # y
        t3 = (aabb[1] - rays_o[..., 1:2]) * dir_fraction[..., 1:2]
        t4 = (aabb[4] - rays_o[..., 1:2]) * dir_fraction[..., 1:2]
        # z
        t5 = (aabb[2] - rays_o[..., 2:3]) * dir_fraction[..., 2:3]
        t6 = (aabb[5] - rays_o[..., 2:3]) * dir_fraction[..., 2:3]

        nears = torch.max(
            torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=-1), dim=-1
        ).values
        fars = torch.min(
            torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=-1), dim=-1
        ).values

        # clamp to near plane
        nears = torch.clamp(nears, min=self.near_plane)
        # assert torch.all(nears < fars), "not collide with scene box"
        valid_mask = nears < fars
        fars = torch.maximum(fars, nears + 1e-6)

        return nears, fars, valid_mask

    def set_nears_and_fars(self, ray_bundle):
        """Intersects the rays with the scene box and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.
        Returns:
            nears: (num_rays, 1)
            fars: (num_rays, 1)
        """
        nears, fars, valid_mask = self._intersect_with_aabb(ray_bundle.origins, ray_bundle.directions, self.scene_box)
        ray_bundle.nears = nears[..., None]
        ray_bundle.fars = fars[..., None]
        if not torch.all(valid_mask):
            ray_bundle.filter(valid_mask)
        return ray_bundle


class NearFarCollider(SceneCollider):
    """Sets the nears and fars with fixed values.

    Args:
        near_plane: distance to near plane
        far_plane: distance to far plane
    """

    def __init__(self, near_plane, far_plane, **kwargs):
        super().__init__(**kwargs)
        self.near_plane = near_plane
        self.far_plane = far_plane
        
    def set_nears_and_fars(self, ray_bundle):
        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        ray_bundle.nears = ones * self.near_plane
        ray_bundle.fars = ones * self.far_plane
        return ray_bundle
