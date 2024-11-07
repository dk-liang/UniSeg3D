import torch
import numpy as np

from mmengine.logging import MMLogger

from mmdet3d.evaluation import InstanceSegMetric
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from .instance_seg_eval import instance_seg_eval

from typing import Dict, Optional, Sequence


def get_iou(gt_masks, pred_masks):
    gt_masks = gt_masks
    num_insts, num_points = gt_masks.shape
    intersection = (gt_masks & pred_masks).reshape(num_insts, num_points).sum(-1)
    union = (gt_masks | pred_masks).reshape(num_insts, num_points).sum(-1)
    ious = (intersection / (union + 1.e-8))
    return ious


@METRICS.register_module()
class PromptSupportedSegMetric(SegMetric):
    def __init__(self,):
        super(PromptSupportedSegMetric, self).__init__()

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            if 'point_prompt_pred_pts_seg' in data_sample:
                point_prompt_pred_3d = data_sample['point_prompt_pred_pts_seg']
                cpu_point_prompt_pred_3d = dict()
                for k, v in point_prompt_pred_3d.items():
                    if hasattr(v, 'to'):
                        cpu_point_prompt_pred_3d[k] = v.to('cpu').numpy()
                    else:
                        cpu_point_prompt_pred_3d[k] = v
                cpu_point_prob = data_sample['point_prob'].to('cpu').numpy()
            
            if 'text_prompt_pred_pts_seg' in data_sample:
                text_prompt_pred_3d = data_sample['text_prompt_pred_pts_seg']
                cpu_text_prompt_pred_3d = dict()
                for k, v in text_prompt_pred_3d.items():
                    if hasattr(v, 'to'):
                        cpu_text_prompt_pred_3d[k] = v.to('cpu').numpy()
                    else:
                        cpu_text_prompt_pred_3d[k] = v

            pred_3d = data_sample['pred_pts_seg']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            if 'point_prompt_pred_pts_seg' in data_sample and 'text_prompt_pred_pts_seg' in data_sample:
                self.results.append((eval_ann_info, cpu_pred_3d, cpu_point_prompt_pred_3d, cpu_point_prob, cpu_text_prompt_pred_3d))
            elif 'point_prompt_pred_pts_seg' in data_sample:
                self.results.append((eval_ann_info, cpu_pred_3d, cpu_point_prompt_pred_3d, cpu_point_prob, None))
            elif 'text_prompt_pred_pts_seg' in data_sample:
                self.results.append((eval_ann_info, cpu_pred_3d, None, None, cpu_text_prompt_pred_3d))
            else:
                self.results.append((eval_ann_info, cpu_pred_3d, None, None, None))


@METRICS.register_module()
class PromptSupportedUnifiedSegMetric(PromptSupportedSegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,   
                 inst_mapping,
                 metric_meta,
                 logger_keys=[('miou',),
                              ('all_ap', 'all_ap_50%', 'all_ap_25%'), 
                              ('pq',)],
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.logger_keys = logger_keys
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        label2cat = self.metric_meta['label2cat']
        ignore_index = self.metric_meta['ignore_index']
        classes = self.metric_meta['classes']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)

        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        val_set_prompt_IoUs_point = np.array([])
        val_set_prompt_IoUs_text = np.array([])
        val_set_point_prob = np.array([])
        for eval_ann, single_pred_results, single_point_prompt_pred_results, single_point_prob, single_text_prompt_pred_results in results:
            if single_point_prompt_pred_results is not None:
                point_prompt_object_gt_masks = single_point_prompt_pred_results['point_prompt_object_gt_masks']
                point_prompt_object_pred_masks = single_point_prompt_pred_results['point_prompt_object_pred_masks']
                if len(point_prompt_object_gt_masks)==0:
                    val_set_prompt_IoUs_point = np.append(val_set_prompt_IoUs_point,np.array([]))
                    val_set_point_prob = np.append(val_set_point_prob, np.array([]))
                else:
                    cur_sample_prompt_IoUs_point = get_iou(point_prompt_object_gt_masks, point_prompt_object_pred_masks)
                    val_set_prompt_IoUs_point = np.append(val_set_prompt_IoUs_point,cur_sample_prompt_IoUs_point)
                    val_set_point_prob = np.append(val_set_point_prob, single_point_prob)
            if single_text_prompt_pred_results is not None:
                text_prompt_object_gt_masks = single_text_prompt_pred_results['text_prompt_object_gt_masks']
                text_prompt_object_pred_masks = single_text_prompt_pred_results['text_prompt_object_pred_masks']
                if len(text_prompt_object_gt_masks)==0:
                    val_set_prompt_IoUs_text = np.append(val_set_prompt_IoUs_text,np.array([]))
                else:
                    cur_sample_prompt_IoUs_text = get_iou(text_prompt_object_gt_masks, text_prompt_object_pred_masks)
                    val_set_prompt_IoUs_text = np.append(val_set_prompt_IoUs_text,cur_sample_prompt_IoUs_text)
            
            gt_masks_pan.append(eval_ann)
            
            pred_masks_pan.append({
                'pts_instance_mask': \
                    single_pred_results['pts_instance_mask'][1],
                'pts_semantic_mask': \
                    single_pred_results['pts_semantic_mask'][1]
            })

            gt_semantic_masks_sem_task.append(eval_ann['pts_semantic_mask'])            
            pred_semantic_masks_sem_task.append(
                single_pred_results['pts_semantic_mask'][0])

            sem_mask, inst_mask = self.map_inst_markup(
                eval_ann['pts_semantic_mask'].copy(), 
                eval_ann['pts_instance_mask'].copy(), 
                self.valid_class_ids[num_stuff_cls:],
                num_stuff_cls)
            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)           
            
            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results['pts_instance_mask'][0]))
            pred_instance_labels.append(
                torch.tensor(single_pred_results['instance_labels']))
            pred_instance_scores.append(
                torch.tensor(single_pred_results['instance_scores']))


        ret_pan = panoptic_seg_eval(
            gt_masks_pan, pred_masks_pan, classes, thing_classes,
            stuff_classes, self.min_num_points, self.id_offset,
            label2cat, ignore_index, logger)

        ret_sem = seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index[0],
            logger=logger)

        # :-1 for unlabeled
        ret_inst = instance_seg_eval(
            gt_semantic_masks_inst_task,
            gt_instance_masks_inst_task,
            pred_instance_masks_inst_task,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids[num_stuff_cls:],
            class_labels=classes[num_stuff_cls:-1],
            logger=logger)

        metrics = dict()
        for ret, keys in zip((ret_sem, ret_inst, ret_pan), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        if single_point_prompt_pred_results is not None:
            metrics.update({'Point Prompt mIoU':np.mean(val_set_prompt_IoUs_point)})
            ap_50 = self.cal_ap_thr(val_set_prompt_IoUs_point, val_set_point_prob, 0.5)
            ap_25 = self.cal_ap_thr(val_set_prompt_IoUs_point, val_set_point_prob, 0.25)
            ap = self.cal_ap(val_set_prompt_IoUs_point, val_set_point_prob)
            metrics.update({'Point Prompt AP': ap,
                            'Point Prompt AP@50': ap_50,
                            'Point Prompt AP@25': ap_25})
        if single_text_prompt_pred_results is not None:
            pre_half = (val_set_prompt_IoUs_text > 0.5).sum() / len(val_set_prompt_IoUs_text)
            pre_quarter = (val_set_prompt_IoUs_text > 0.25).sum() / len(val_set_prompt_IoUs_text)
            metrics.update({'Text Prompt mIoU':np.mean(val_set_prompt_IoUs_text),
                            'Text_Precision_half': pre_half,
                            'Text_Precision_quarter': pre_quarter})
        return metrics
    
    def cal_ap_thr(self, ious, confidences, thr=0.5):
        true_positives = ious > thr
        sorted_indices = np.argsort(-confidences)
        confidences = confidences[sorted_indices]
        true_positives = true_positives[sorted_indices]

        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(~true_positives)
        total_positives = len(true_positives)

        precisions = cumulative_tp / (cumulative_tp + cumulative_fp)
        recalls = cumulative_tp / total_positives

        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))
        mpre_inds = range(len(mpre) - 2, -1, -1)
        for i in mpre_inds:
            mpre[i] = max(mpre[i], mpre[i + 1])
        ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])
        return ap

    def cal_ap(self, ious, condidences):
        ap = 0
        for i in range(10):
            thr = 0.5 + 0.05*i
            ap += self.cal_ap_thr(ious, condidences, thr)
        ap /= 10.
        return ap
    
    def map_inst_markup(self,
                        pts_semantic_mask,
                        pts_instance_mask,
                        valid_class_ids,
                        num_stuff_cls):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes
        
        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]
        
        return pts_semantic_mask, pts_instance_mask
