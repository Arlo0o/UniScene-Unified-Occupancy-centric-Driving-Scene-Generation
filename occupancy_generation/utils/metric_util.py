import numpy as np
from mmengine import MMLogger
logger = MMLogger.get_instance('genocc')
import torch
import torch.distributed as dist
from scipy import linalg
from sklearn import metrics
import torch.nn.functional as F

class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name
                 # empty_class: int
        ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()

    def _after_epoch(self):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        logger = MMLogger.get_current_instance()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('%s : %.2f%%' % (label_str, iou * 100))
        
        return miou * 100


class multi_step_MeanIou:
    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name,
                 times=1):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name
        self.times = times
        
    def reset(self) -> None:
        self.total_seen = torch.zeros(self.times, self.num_classes).cuda()
        self.total_correct = torch.zeros(self.times, self.num_classes).cuda()
        self.total_positive = torch.zeros(self.times, self.num_classes).cuda()
    
    def _after_step(self, outputses, targetses):
        
        assert outputses.shape[1] == self.times, f'{outputses.shape[1]} != {self.times}'
        assert targetses.shape[1] == self.times, f'{targetses.shape[1]} != {self.times}'
        for t in range(self.times):
            outputs = outputses[:,t, ...][targetses[:,t, ...] != self.ignore_label].cuda()
            targets = targetses[:,t, ...][targetses[:,t, ...] != self.ignore_label].cuda()
            for j, c in enumerate(self.class_indices):
                self.total_seen[t, j] += torch.sum(targets == c).item()
                self.total_correct[t, j] += torch.sum((targets == c)
                                                      & (outputs == c)).item()
                self.total_positive[t, j] += torch.sum(outputs == c).item()
    
    def _after_epoch(self):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
        mious = []
        for t in range(self.times):
            ious = []
            for i in range(self.num_classes):
                if self.total_seen[t, i] == 0:
                    ious.append(1)
                else:
                    cur_iou = self.total_correct[t, i] / (self.total_seen[t, i]
                                                          + self.total_positive[t, i]
                                                          - self.total_correct[t, i])
                    ious.append(cur_iou.item())
            miou = np.mean(ious)
            logger = MMLogger.get_current_instance()
            logger.info(f'per class iou {self.name} at time {t}:')
            for iou, label_str in zip(ious, self.label_str):
                logger.info('%s : %.2f%%' % (label_str, iou * 100))
            logger.info(f'mIoU {self.name} at time {t}: %.2f%%' % (miou * 100))
            mious.append(miou * 100)
        return mious, np.mean(mious)


class multi_step_MeanIou_woD:
    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name,
                 times=1):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name
        self.times = times
        
    def reset(self) -> None:
        self.total_seen = torch.zeros(self.times, self.num_classes).cuda()
        self.total_correct = torch.zeros(self.times, self.num_classes).cuda()
        self.total_positive = torch.zeros(self.times, self.num_classes).cuda()
    
    def _after_step(self, outputses, targetses):
        
        assert outputses.shape[1] == self.times, f'{outputses.shape[1]} != {self.times}'
        assert targetses.shape[1] == self.times, f'{targetses.shape[1]} != {self.times}'
        for t in range(self.times):
            outputs = outputses[:,t, ...][targetses[:,t, ...] != self.ignore_label].cuda()
            targets = targetses[:,t, ...][targetses[:,t, ...] != self.ignore_label].cuda()
            for j, c in enumerate(self.class_indices):
                self.total_seen[t, j] += torch.sum(targets == c).item()
                self.total_correct[t, j] += torch.sum((targets == c)
                                                      & (outputs == c)).item()
                self.total_positive[t, j] += torch.sum(outputs == c).item()
    
    def _after_epoch(self,logfile):
        mious = []
        for t in range(self.times):
            ious = []
            for i in range(self.num_classes):
                if self.total_seen[t, i] == 0:
                    ious.append(1)
                else:
                    cur_iou = self.total_correct[t, i] / (self.total_seen[t, i]
                                                          + self.total_positive[t, i]
                                                          - self.total_correct[t, i])
                    ious.append(cur_iou.item())
            miou = np.mean(ious)
            for iou, label_str in zip(ious, self.label_str):
                with open(logfile, 'a') as file:
                    print('%s : %.2f%%' % (label_str, iou * 100),file=file)
            with open(logfile, 'a') as file:
                print(f'mIoU {self.name} at time {t}: %.2f%%' % (miou * 100),file=file)
            mious.append(miou * 100)
        return mious, np.mean(mious)

class multi_step_fid_mmd:
    def __init__(self,feature_shape=2048,kernel_bandwidth=1.0):
        self.feature_shape = feature_shape
        self.kernel_bandwidth = kernel_bandwidth
        self.features_ori = []
        self.features_gen = []


    def _after_step(self, feature_ori, feature_gen):
        feature_ori = self.gather_features(feature_ori)
        feature_gen = self.gather_features(feature_gen)

        self.features_ori.append(feature_ori)
        self.features_gen.append(feature_gen)
        # print(len(self.features_gen))

    def _after_epoch(self,logfile=None):
        features_ori = torch.cat(self.features_ori, dim=0).cpu().numpy()
        features_gen = torch.cat(self.features_gen, dim=0).cpu().numpy()

        mu1, sigma1 = np.mean(features_ori, axis=0), np.cov(features_ori, rowvar=False)
        mu2, sigma2 = np.mean(features_gen, axis=0), np.cov(features_gen, rowvar=False)
        # print(mu1.shape,sigma1.shape)
        fid_value = self.calculate_fid(mu1, sigma1, mu2, sigma2)
        # print(fid_value)
        mmd_value = self.calculate_mmd(features_ori, features_gen)

        return fid_value,mmd_value
    
    @staticmethod
    def calculate_fid(mu1, sigma1, mu2, sigma2):

        diff = mu1 - mu2


        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def calculate_mmd(self, features_ori, features_gen):
        
        def median_heuristic(X, Y=None):
            if Y is None:  
                dists = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=-1)
            else:
                dists = np.sum((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=-1)
            return np.median(dists)


        def gaussian_kernel(x, y, bandwidth):
            dist = np.sum((x[:, np.newaxis] - y[np.newaxis, :]) ** 2, axis=-1)
            return np.exp(-dist / (2 * bandwidth ** 2))

        # self.kernel_bandwidth = median_heuristic(features_ori, features_gen)
        # print(self.kernel_bandwidth)

        # K_oo = gaussian_kernel(features_ori, features_ori, self.kernel_bandwidth)  
        # K_gg = gaussian_kernel(features_gen, features_gen, self.kernel_bandwidth)  
        # K_og = gaussian_kernel(features_ori, features_gen, self.kernel_bandwidth)  
        # mmd_value = K_oo.mean() + K_gg.mean() - 2 * K_og.mean()

        # N = features_ori.shape[0]
        # bandwidth = self.kernel_bandwidth
        # batch_size=500
        # mmd_value = 0
        # for i in range(0, N, batch_size):
        #     ori_batch = features_ori[i:i+batch_size]
        #     for j in range(0, N, batch_size):
        #         gen_batch = features_gen[j:j+batch_size]
                
        #         K_oo = gaussian_kernel(ori_batch, ori_batch, bandwidth)
        #         K_gg = gaussian_kernel(gen_batch, gen_batch, bandwidth)
        #         K_og = gaussian_kernel(ori_batch, gen_batch, bandwidth)
                
        #         mmd_value += K_oo.mean() + K_gg.mean() - 2 * K_og.mean()
        # mmd_value /= (N // batch_size) ** 2

        # return mmd_value

        X = features_ori
        Y = features_gen
        gamma = 1.0 / (2.0 * self.kernel_bandwidth ** 2)
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

        return XX.mean() + YY.mean() - 2 * XY.mean()

    def gather_features(self, features):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return features
            
        world_size = dist.get_world_size() 
        gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
        
        dist.all_gather(gathered_features, features)
        
        return torch.cat(gathered_features, dim=0)


class multi_step_TemporalConsistency:
    def __init__(self,
                 name,
                 times=1):
        self.name = name
        self.times = times
        
    def reset(self) -> None:
        self.total_similarity = torch.tensor(0.0).cuda()
        self.total_count = torch.tensor(0.0).cuda()

    def _after_step(self, feature_batches):
        """
        feature_batches: Tensor with shape (B, T, 2048)
        """
        assert feature_batches.shape[1] == self.times, f'{feature_batches.shape[1]} != {self.times}'
        
        first_frame_features = None
        former_frame_features = None
        
        for t in range(self.times):
            # Normalization across the last dimension (2048)
            normalized_features = F.normalize(feature_batches[:, t, :], dim=-1, p=2).cuda()

            # For time t=0, we store the features to compare with subsequent frames
            if t == 0:
                first_frame_features = normalized_features
                former_frame_features = normalized_features
            else:
                # Cosine similarity between current and previous frame
                sim_pre = F.cosine_similarity(former_frame_features, normalized_features, dim=-1)
                
                # Cosine similarity between current frame and the first frame
                sim_fir = F.cosine_similarity(first_frame_features, normalized_features, dim=-1)
                
                # Averaging the two similarities
                cur_sim = (sim_pre + sim_fir) / 2.0
                # print(cur_sim.shape)

                # Summing similarity for this time step
                self.total_similarity += torch.sum(cur_sim).item()
                self.total_count += cur_sim.shape[0]  # Add the number of samples (batch size)

                # Update former frame for the next iteration
                former_frame_features = normalized_features

    def _after_epoch(self):
        # Reduce across all processes (in multi-GPU training)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(self.total_similarity)
            dist.all_reduce(self.total_count)

        # Compute the average temporal consistency per time step
        temporal_consistencies = self.total_similarity / self.total_count
        # for t in range(1, self.times):  # Skip t=0 since there's no comparison for the first frame
        #     if self.total_count[t] == 0:
        #         consistency = 1.0  # Default to perfect consistency if no frames are available
        #     else:
        #         consistency = self.total_similarity[t] / self.total_count[t]
        #     temporal_consistencies.append(consistency)

        return temporal_consistencies
