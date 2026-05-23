import numpy as np
import logging
# from ssc_metric import SSCMetrics, nuscenes_class_names 
from copy import deepcopy
from tqdm import tqdm

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the class indices and label strings
carla_class_names = [
  'Free',
  'Building',
  'Barrier',
  'Other',
  'Pedestrian',
  'Pole',
  'Road',
  'Ground',
  'Sidewalk',
  'Vegetation',
  'Vehicle',
]

nuscenes_class_names = [
        'free',
        'barrier',
        'bicycle',
        'bus',
        'car',
        'construction_vehicle',
        'motorcycle',
        'pedestrian',
        'traffic_cone',
        'trailer',
        'truck',
        'driveable_surface',
        'other_flat',
        'sidewalk',
        'terrain',
        'manmade',
        'vegetation',
    ]

# ================Occworld Metrics====Start============
# Part of the code is taken from https://github.com/wzzheng/OccWorld/blob/main/utils/metric_util.py
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
        self.total_seen = np.zeros((self.times, self.num_classes))
        self.total_correct = np.zeros((self.times, self.num_classes))
        self.total_positive = np.zeros((self.times, self.num_classes))
    
    def _after_step(self, outputses, targetses):
        
        assert outputses.shape[1] == self.times, f'{outputses.shape[1]} != {self.times}'
        assert targetses.shape[1] == self.times, f'{targetses.shape[1]} != {self.times}'
        for t in range(self.times):
            outputs = outputses[:,t, ...][targetses[:,t, ...] != self.ignore_label]
            targets = targetses[:,t, ...][targetses[:,t, ...] != self.ignore_label]
            for j, c in enumerate(self.class_indices):
                self.total_seen[t, j] += np.sum(targets == c)
                self.total_correct[t, j] += np.sum((targets == c) & (outputs == c))
                self.total_positive[t, j] += np.sum(outputs == c)
    
    def _after_epoch(self):
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
                    ious.append(cur_iou)
            miou = np.mean(ious)
            logger.info(f'per class iou {self.name} at time {t}:')
            for iou, label_str in zip(ious, self.label_str):
                logger.info('%s : %.2f%%' % (label_str, iou * 100))
            logger.info(f'mIoU {self.name} at time {t}: %.2f%%' % (miou * 100))
            mious.append(miou * 100)
        return mious, np.mean(mious)
    
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
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += np.sum(targets == c) # 
            self.total_correct[i] += np.sum((targets == c) & (outputs == c)) # TP
            self.total_positive[i] += np.sum(outputs == c) # TP + FP

    def _after_epoch(self):
        # dist.all_reduce(self.total_seen)
        # dist.all_reduce(self.total_correct)
        # dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]+ self.total_positive[i] - self.total_correct[i])
                ious.append(cur_iou)

        miou = np.mean(ious)

        logger.info(f'Validation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('%s : %.2f%%' % (label_str, iou * 100))
        
        return miou * 100
# ================Occworld Metrics=====End===========

# ================SSC Metrics Start ================
def get_iou(iou_sum, cnt_class):
    _C = iou_sum.shape[0]  # 12
    iou = np.zeros(_C, dtype=np.float32)  # iou for each class
    for idx in range(_C):
        iou[idx] = iou_sum[idx] / cnt_class[idx] if cnt_class[idx] else 0

    mean_iou = np.sum(iou[1:]) / np.count_nonzero(cnt_class[1:])
    return iou, mean_iou

def get_accuracy(predict, target, weight=None):  # 0.05s
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]  # _C = 12
    # target = np.int32(target)
    target = target.reshape(_bs,_C, -1)  # (_bs, 60*36*60) 129600
    predict = predict.reshape(_bs, _C, -1)  # (_bs, _C, 60*36*60)
    # predict = np.argmax(predict, axis=1)  # one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.

    correct = predict == target  # (_bs, 129600)
    if weight is not None:  # 0.04s, add class weights
        weight_k = np.ones(target.shape)
        for i in range(_bs):
            for n in range(target.shape[1]):
                idx = 0 if target[i, n] == 255 else target[i, n]
                weight_k[i, n] = weight[idx]
        correct = correct * weight_k
    acc = correct.sum() / correct.size
    return acc
class SSCMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def hist_info(self, n_cl, pred, gt):
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        mean_IU_no_back = np.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):
        self.count += 1
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(y_pred, y_true, mask)
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

    def get_stats(self):
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (self.completion_tp + self.completion_fp + self.completion_fn)
        else:
            precision, recall, iou = 0, 0, 0
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean_w_empty": np.mean(iou_ssc[iou_ssc != 0]),
            "iou_ssc_mean_wn_empty": np.mean(iou_ssc[1:][iou_ssc[1:] != 0]),
        }

    def reset(self):
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)

    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
                y_true = y_true[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
            for j in range(_C):  # for each class
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum


if __name__ == '__main__':
    import os
    # test
    old_metrics = SSCMetrics(n_classes=17)

    unique_label = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
    # ['free', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
    unique_label_str=['free', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']

    CalMeanIou_sem = multi_step_MeanIou(unique_label, -100, unique_label_str, 'sem',times=8)
    CalMeanIou_vox = multi_step_MeanIou([1], -100, ['occupied'], 'vox', times=8)

    # CalMeanIou_sem = MeanIoU(unique_label, -100, unique_label_str, 'sem')
    # CalMeanIou_vox = MeanIoU([1], -100, ['occupied'], 'vox')

    CalMeanIou_sem.reset()
    CalMeanIou_vox.reset()

    gt_occ_path = '/code/code/Diff_occ/occ_gen/occ_gen/out/eval_12hz_VAE/1226_metrics/input'
    pred_occ_path = '/code/code/Diff_occ/occ_gen/occ_gen/out/eval_12hz_VAE/1226_metrics/reconsem'

    gt_occ_files = []
    for root, dirs, files in os.walk(gt_occ_path):
        for filename in files:
            if filename.endswith('.npy'):
                gt_occ_files.append(os.path.join(root, filename))
    pred_occ_files = []

    for root, dirs, files in os.walk(pred_occ_path):
        for filename in files:
            if filename.endswith('.npy'):
                pred_occ_files.append(os.path.join(root, filename))
    gt_occ_files.sort()
    pred_occ_files.sort()
    count = 1
    os.makedirs('/code/code/Diff_occ/occ_gen/occ_gen/out/eval_12hz_VAE/metrics', exist_ok=True)
    # os.('/code/code/Diff_occ/occ_gen/occ_gen/out/eval_12hz_VAE/metrics', exist_ok=True)
    total_files = min(len(gt_occ_files), len(pred_occ_files))

    # for gt_occ_file, pred_occ_file in zip(gt_occ_files, pred_occ_files):
    for gt_occ_file, pred_occ_file in tqdm(zip(gt_occ_files, pred_occ_files), total=total_files):
        
        # print(f"Processing {gt_occ_file} and {pred_occ_file}")
        gt_occ = np.load(gt_occ_file)
        pred_occ = np.load(pred_occ_file)


        pred_occs_iou = deepcopy(pred_occ)
        pred_occs_iou_vox = deepcopy(pred_occ)
        pred_occs_iou[pred_occs_iou == 17] = 0
        pred_occs_iou_vox[pred_occs_iou != 0] = 1

        target_occs_iou = deepcopy(gt_occ)
        target_occs_iou_vox = deepcopy(gt_occ)
        target_occs_iou[target_occs_iou == 17] = 0
        target_occs_iou_vox[target_occs_iou != 0] = 1

        CalMeanIou_sem._after_step(pred_occs_iou, target_occs_iou)
        CalMeanIou_vox._after_step(pred_occs_iou_vox, target_occs_iou_vox)
        old_metrics.add_batch(pred_occs_iou, target_occs_iou)

        
        if count<10:
            np.save(f'/code/code/Diff_occ/occ_gen/occ_gen/out/eval_12hz_VAE/metrics/pred_vox_{count}', pred_occs_iou_vox)
            np.save('/code/code/Diff_occ/occ_gen/occ_gen/out/eval_12hz_VAE/metrics/target_vox_{count}', target_occs_iou_vox)
            np.save('/code/code/Diff_occ/occ_gen/occ_gen/out/eval_12hz_VAE/metrics/pred_sem_{count}', pred_occs_iou)
            np.save('/code/code/Diff_occ/occ_gen/occ_gen/out/eval_12hz_VAE/metrics/target_sem_{count}', target_occs_iou)
        count += 1
        # for i in range(255):
        #     if np.any(target_occs_iou==i) or np.any(pred_occs_iou==i):
        #         print(f"target {i}: {np.any(target_occs_iou==i)}")
        #         print(f"pred {i}: {np.any(pred_occs_iou==i)}")

    # pred_occs = np.load('/Users/hu/code/occ-vis/data/test/6_input/6_pred.npy')

    # target_occs = np.load('/Users/hu/code/occ-vis/data/test/6_input/6_target.npy')
    # # print if pred or target include 17


    # pred_occs_iou = deepcopy(pred_occs)
    # pred_occs_iou[pred_occs_iou != 0] = 1
    # target_occs_iou = deepcopy(target_occs)
    # target_occs_iou[target_occs_iou != 0] = 1

    # CalMeanIou_sem._after_step(pred_occs, target_occs)
    # CalMeanIou_vox._after_step(pred_occs_iou, target_occs_iou)
    # old_metrics_stats = old_metrics.add_batch(pred_occs, target_occs)

    best_val_iou = [0]*10
    best_val_miou = [0]*10

    val_miou = CalMeanIou_sem._after_epoch()
    val_iou = CalMeanIou_vox._after_epoch()
    old_metrics_stats = old_metrics.get_stats()

    # best_val_iou = [max(best_val_iou[i], val_iou[i]) for i in range(len(best_val_iou))]
    # best_val_miou = [max(best_val_miou[i], val_miou[i]) for i in range(len(best_val_miou))]
    logger.info(f'Current val vox iou is {val_iou} ')
    logger.info(f'Current val sem miou is {val_miou} ')

    logger.info(f'\n Validation Metrics at epoch ======')
    logger.info("Precision={:.2f}%, Recall={:.2f}%, IoU={:.2f}%".format(
        old_metrics_stats["precision"] * 100, old_metrics_stats["recall"] * 100, old_metrics_stats["iou"] * 100
    ))
    logger.info("Class IoU:")
    for class_name, iou_value in zip(nuscenes_class_names, old_metrics_stats["iou_ssc"]):
        logger.info("  {:15s}: {:.2f}%".format(class_name, iou_value * 100))
    print("Mean IoU (with 'empty' class): {:.2f}%".format(old_metrics_stats["iou_ssc_mean_w_empty"] * 100))
    print("Mean IoU (w/o 'empty' class): {:.2f}%".format(old_metrics_stats["iou_ssc_mean_wn_empty"] * 100))