from .base_surface_model import SurfaceModel
from functools import partial


class VolSDFModel(SurfaceModel):
    def __init__(
        self,
        pc_range,
        voxel_size,
        voxel_shape,
        field_cfg,
        collider_cfg,
        sampler_cfg,
        loss_cfg,
        norm_scene,
        **kwargs
    ):
        super().__init__(
            pc_range=pc_range,
            voxel_size=voxel_size,
            voxel_shape=voxel_shape,
            field_cfg=field_cfg,
            collider_cfg=collider_cfg,
            sampler_cfg=sampler_cfg,
            loss_cfg=loss_cfg,
            norm_scene=norm_scene,
            **kwargs
        )

    def sample_and_forward_field(self, ray_bundle, feature_volume):
        sampler_out_dict = self.sampler(
            ray_bundle,
            density_fn=self.field.laplace_density,
            sdf_fn=partial(self.field.get_sdf, feature_volume=feature_volume),
        )
        ray_samples = sampler_out_dict.pop("ray_samples")
        field_outputs = self.field(ray_samples, feature_volume)
        weights, _ = ray_samples.get_weights_and_transmittance(field_outputs["density"])

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "sampled_points": ray_samples.frustums.get_start_positions(),
            **sampler_out_dict,
        }
        return samples_and_field_outputs
