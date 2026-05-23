from mmengine.registry import MODELS
from mmengine.model import BaseModule
import numpy as np
import torch.nn as nn, torch
import torch.nn.functional as F
from einops import rearrange
from copy import deepcopy
import torch.distributions as dist
from utils.metric_stp3 import PlanningMetric
import time

@MODELS.register_module()
class TransVQVAE(BaseModule):
    def __init__(self, vae, transformer, num_frames=10, offset=1,
                 pose_encoder=None, pose_decoder=None,
                 pose_actor=None, give_hiddens=False, delta_input=False, without_all=False):
        super().__init__()
        self.num_frames = num_frames
        self.offset = offset
        self.vae = MODELS.build(vae)
        self.transformer = MODELS.build(transformer)
        if pose_encoder is not None:
            self.pose_encoder = MODELS.build(pose_encoder)
        if pose_decoder is not None:
            self.pose_decoder = MODELS.build(pose_decoder)
        if pose_actor is not None:
            self.pose_actor = MODELS.build(pose_actor)
        self.give_hiddens = give_hiddens
        self.delta_input = delta_input
        self.planning_metric = None
        self.without_all = without_all
    def forward(self, x, metas=None):
        if hasattr(self, 'pose_encoder'):
            if self.training:
                return self.forward_train_with_plan(x, metas)
            else:
                return self.forward_inference_with_plan(x, metas)
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x)
    def forward_train(self, x):
        # given x: bs, f, h, w, d where f == num_frames + offset
        # output : ce_inputs: logits for the codebook 
        # output : ce_labels: labels for the ce_inputs
        assert hasattr(self.vae, 'vqvae')
        bs, F, H, W, D = x.shape
        assert F == self.num_frames + self.offset
        output_dict = {}
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        # z: bs*f, c, h, w 
        
        # z: bs*f, h, w
        return output_dict
        
    
    def forward_inference(self, x):
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        z_q_predict = z_q_predict.argmax(dim=1)
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
        
        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou
    
        return output_dict
    
    
    def forward_train_with_plan(self, x, metas):
        assert hasattr(self.vae, 'vqvae')
        assert hasattr(self, 'pose_encoder')
        bs, F, H, W, D = x.shape
        assert F == self.num_frames + self.offset
        output_dict = {}
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
            
            
        rel_poses, output_metas = self._get_pose_feature(metas, F-self.offset)
        
        z_q_predict, rel_poses = self.transformer(z_q[:, :self.num_frames], pose_tokens=rel_poses)
        
        pose_decoded = self.pose_decoder(rel_poses)
        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas
        
        
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        # z: bs*f, c, h, w 
        
        # z: bs*f, h, w
        return output_dict
    def forward_inference_with_plan(self, x, metas):
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
            
        
        rel_poses, output_metas = self._get_pose_feature(metas, F-self.offset)
        
        z_q_predict, rel_poses = self.transformer(z_q[:, :self.num_frames], pose_tokens=rel_poses)
        
        pose_decoded = self.pose_decoder(rel_poses)
        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas
        
        
        
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        z_q_predict = z_q_predict.argmax(dim=1)
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
        
        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou
    
        return output_dict

    def _get_pose_feature(self, metas=None, F=None):
        rel_poses, output_metas = None, None
        if hasattr(self, 'pose_encoder'):
            assert hasattr(self, 'pose_decoder')
            assert metas is not None
            output_metas = []
            for meta in metas:
                output_meta = dict()
                output_meta['rel_poses'] = meta['rel_poses'][self.offset:]
                output_meta['gt_mode'] = meta['gt_mode'][self.offset:]
                output_metas.append(output_meta)
                
                
            rel_poses = np.array([meta['rel_poses'] for meta in metas])
            gt_mode = np.array([meta['gt_mode'] for meta in metas])
            
            
            
            
            gt_mode = torch.tensor(gt_mode).cuda()
            rel_poses = torch.tensor(rel_poses).cuda()# list of (num_frames+offsets, 2)
            if self.delta_input:
                rel_poses_pre = torch.cat([torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1]], dim=1)
                rel_poses = rel_poses - rel_poses_pre
            if F>self.num_frames:
                assert F == self.num_frames + self.offset
            else:
                assert F == self.num_frames
                gt_mode = gt_mode[:, :-self.offset, :] 
                rel_poses = rel_poses[:, :-self.offset, :]
                
            rel_poses = torch.cat([rel_poses, gt_mode], dim=-1)
            #rel_poses = rearrange(rel_poses, 'b f d -> b f 1 d')
            rel_poses = self.pose_encoder(rel_poses.float())
        return rel_poses, output_metas
    
    def forward_autoreg_with_pose(self, x, metas, start_frame=0, mid_frame=6,end_frame=12):
        t0 = time.time()
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['input_occs'] = x[:, mid_frame-1:end_frame]
        output_dict['target_occs'] = x[:, mid_frame:end_frame]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, mid_frame:end_frame].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        z_q_predict = z_q[:, start_frame:mid_frame]
        t1 = time.time()
        output_metas = []
        input_metas = []
        for meta in metas:
            input_meta = dict()
            input_meta['rel_poses'] = meta['rel_poses'][start_frame:mid_frame]
            input_meta['gt_mode'] = meta['gt_mode'][start_frame:mid_frame]
            input_metas.append(input_meta)
        output_dict['input_metas'] = input_metas
        for meta in metas:
            output_meta = dict()
            output_meta['rel_poses'] = meta['rel_poses'][mid_frame:end_frame]#-meta['rel_poses'][mid_frame-1]
            output_meta['gt_mode'] = meta['gt_mode'][mid_frame:end_frame]
            output_metas.append(output_meta)
        output_dict['gt_poses_'] = np.array([meta['rel_poses'] for meta in output_metas])
        rel_poses = np.array([meta['rel_poses'] for meta in metas])
        gt_mode = np.array([meta['gt_mode'] for meta in metas])
        gt_mode = torch.tensor(gt_mode).cuda()
        
        rel_poses = torch.tensor(rel_poses).cuda()
        if self.delta_input:
            rel_poses_pre = torch.cat(torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1], dim=1)
            rel_poses = rel_poses - rel_poses_pre
        rel_poses_sumed = rel_poses[:, start_frame:mid_frame]
        rel_poses = torch.cat([rel_poses, gt_mode], dim=-1)
        rel_poses = rel_poses[:, start_frame:mid_frame]
        
        rel_poses = self.pose_encoder(rel_poses.float())
        rel_poses_state = rel_poses
        z_q_list = []
        t2 = time.time()
        poses_ = []
        for i in range(mid_frame, end_frame):
            z_q_, rel_poses_ = self.transformer.forward_autoreg_step(
                z_q_predict, pose_tokens=rel_poses_state,
                start_frame=start_frame, mid_frame=i)
            z_q_list.append(z_q_[:, -1:])
            #print(z_q_.shape)
            z_q_ = z_q_[:, -1:].clone().detach().argmax(dim=2)
            #print(z_q_.shape)
            z_q_ = self.vae.vqvae.get_codebook_entry(z_q_, shape=None)
            z_q_ = rearrange(z_q_, 'b f h w c-> b f c h w')
            z_q_predict = torch.cat([z_q_predict, z_q_], dim=1)
            rel_poses = torch.cat([rel_poses, rel_poses_[:, -1:]], dim=1)
            rel_poses_state_, rel_poses_sumed, pose_ = self.decode_pose(rel_poses_[:, -1:], gt_mode[:,i:i+1], rel_poses_sumed)
            poses_.append(pose_)
            rel_poses_state = torch.cat([rel_poses_state, rel_poses_state_], dim=1)
        poses_ = torch.cat(poses_, dim=1)
        output_dict['poses_'] = poses_
        t3 = time.time()
        z_q_predict = z_q_predict[:, mid_frame:end_frame]
        rel_poses = rel_poses[:, mid_frame:end_frame]
        #assert False, f'z_q_predict.shape: {z_q_predict.shape}, rel_poses.shape: {rel_poses.shape}, {output_dict["target_occs"].shape}'
        # print(z_q_predict.shape, rel_poses.shape)
        pose_decoded = self.pose_decoder(rel_poses)
        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas
        
        z_q = torch.cat(z_q_list, dim=1)
        #print(z_q.shape)
        output_dict['ce_inputs'] = z_q.flatten(0, 1)
        z_q_predict = z_q_predict.flatten(0, 1)
        # output_dict['ce_inputs'] = z_q_predict
        # z_q_predict = z_q_predict.argmax(dim=1)
        # z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        # z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
        
        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou

        
        if self.without_all:
            #output_dict['pose_decoded'] = 
            output_dict['sem_pred'] = output_dict['input_occs'][:, 0:1].repeat(1, end_frame-mid_frame, 1, 1, 1)
            pred_iou = deepcopy(output_dict['sem_pred'])
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou
            output_dict['pose_decoded'] = torch.tensor([meta['rel_poses'] for meta in input_metas])[:,-1:].unsqueeze(2).repeat(1, end_frame-mid_frame, 3, 1)
        output_dict['time'] = {'encode':t1-t0, 'mid':t2-t1, 'autoreg':t3-t2, 'total':t3-t0, 'per_frame':t1-t0+(t3-t2)/(end_frame-mid_frame)}
        return output_dict
        
        
        
    def decode_pose(self, pose, gt_mode, rel_poses_sumed):
        pose = self.pose_decoder(pose)
        # pose:b, f, 3, 2
        # mode:b, f, 3
        # b, f, 2
        bs, num_frames, num_modes, _ = pose.shape
        #gt_mode_ = gt_mode.unsqueeze(-1).repeat(1, 1, 1, 2)
        pose = pose[gt_mode.bool()].reshape(bs, num_frames, 2)
        pose_decoded = pose.clone().detach()
        '''if not self.delta_input:
            pose = pose+rel_poses_sumed[:, -1:]
            rel_poses_sumed = torch.cat([rel_poses_sumed, pose], dim=1)'''
        pose = torch.cat([pose, gt_mode], dim=-1)
        pose = self.pose_encoder(pose.float())
        return pose, rel_poses_sumed, pose_decoded
    def forward_autoreg(self, x, metas=None, start_frame=0, mid_frame=6,end_frame=12):
        
        pass
    def generate_inference(self, x):
        #import pdb; pdb.set_trace()
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        z_q_predict = z_q_predict.permute(0, 2, 3, 1)
        cata_distribution = dist.Categorical(logits=(z_q_predict-z_q_predict.min())/(z_q_predict.max()-z_q_predict.min()))
        import pdb;pdb.set_trace()
        z_q_predict = cata_distribution.sample()
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
        
        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou
    
        return output_dict
    
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3"""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            'plan_L2_1s_single':0,
            'plan_L2_2s_single':0,
            'plan_L2_3s_single':0,
            'plan_obj_col_1s_single':0,
            'plan_obj_col_2s_single':0,
            'plan_obj_col_3s_single':0,
            'plan_obj_box_col_1s_single':0,
            'plan_obj_box_col_2s_single':0,
            'plan_obj_box_col_3s_single':0,
            
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                traj_L2_single = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, cur_time-1:cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, cur_time-1:cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                obj_coll_single, obj_box_coll_single = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, cur_time-1:cur_time].detach(),
                    gt_ego_fut_trajs[:, cur_time-1:cur_time],
                    occupancy[:, cur_time-1:cur_time])
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_L2_{}s_single'.format(i+1)] = traj_L2_single
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
                metric_dict['plan_obj_col_{}s_single'.format(i+1)] = obj_coll_single.item()
                metric_dict['plan_obj_box_col_{}s_single'.format(i+1)] = obj_box_coll_single.item()
                
                
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_L2_{}s_single'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict
    
    def autoreg_for_stp3_metric(self, x, metas, 
                                start_frame=0, mid_frame=6,end_frame=12):
        output_dict = self.forward_autoreg_with_pose(x, metas, start_frame, mid_frame, end_frame)
        pred_ego_fut_trajs = output_dict['pose_decoded']
        gt_mode = torch.tensor([meta['gt_mode'] for meta in output_dict['output_metas']])
        bs, num_frames, num_modes, _ = pred_ego_fut_trajs.shape
        pred_ego_fut_trajs = pred_ego_fut_trajs[gt_mode.bool()].reshape(bs, num_frames, 2)
        pred_ego_fut_trajs = torch.cumsum(pred_ego_fut_trajs, dim=1).cpu()
        gt_ego_fut_trajs = torch.tensor([meta['rel_poses'] for meta in output_dict['output_metas']])
        gt_ego_fut_trajs = torch.cumsum(gt_ego_fut_trajs, dim=1).cpu()
        assert len(metas) == 1, f'len(metas): {len(metas)}'
        gt_bbox = metas[0]['gt_bboxes_3d']
        gt_attr_labels = torch.tensor(metas[0]['attr_labels'])
        fut_valid_flag = torch.tensor(metas[0]['fut_valid_flag'])
        # import pdb;pdb.set_trace()
        metric_stp3 = self.compute_planner_metric_stp3(
            pred_ego_fut_trajs, gt_ego_fut_trajs, 
            gt_bbox, gt_attr_labels[None], True)
        
        output_dict['metric_stp3'] = metric_stp3
        
        return output_dict