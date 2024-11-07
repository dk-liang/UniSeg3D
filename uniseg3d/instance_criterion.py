import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .structures import InstanceData_
from mmdet3d.registry import MODELS, TASK_UTILS


def batch_sigmoid_bce_loss(inputs, targets):
    """Sigmoid BCE loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')

    pos_loss = torch.einsum('nc,mc->nm', pos, targets)
    neg_loss = torch.einsum('nc,mc->nm', neg, (1 - targets))
    return (pos_loss + neg_loss) / inputs.shape[1]


def batch_dice_loss(inputs, targets):
    """Dice loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def get_iou(inputs, targets):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).
    
    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

@MODELS.register_module()
class OVInstanceCriterion:
    def __init__(self, matcher, ov_matcher,  loss_weight, non_object_weight, num_classes,
                 fix_dice_loss_weight, IoU_loss_weight, IntrinsicPointAndQuery_loss_weight, iter_matcher, 
                 ratio=0.1, iou_thr=0., pred_iou=False, fix_mean_loss=False):
        self.matcher = TASK_UTILS.build(matcher)
        self.ov_matcher = TASK_UTILS.build(ov_matcher)
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.pred_iou = pred_iou
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.IoU_loss_weight = IoU_loss_weight
        self.IntrinsicPointAndQuery_loss_weight = IntrinsicPointAndQuery_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        self.ratio = ratio

        self.bce_criterion = nn.BCELoss()
        self.iou_thr = iou_thr

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        """Per layer auxiliary loss.

        Args:
            aux_outputs (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
            insts (List):
                Ground truth of len batch_size, each InstanceData_ with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Tensor: loss value.
        """
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']
        device = pred_masks[0].device
        #Convenient for subsequent processing, value=[None, ...]
        ov_pred_scores = aux_outputs['scores']
        
        if self.pred_iou:
            preds_iou = aux_outputs['preds_iou']
        base_insts = []
        for i in range(len(insts)):
            is_novel = insts[i].is_novel
            base_inst = InstanceData_(
                labels_3d = insts[i].labels_3d[~is_novel],
                sp_masks = insts[i].sp_masks[~is_novel]
            )
            if insts[i].get('query_masks') is not None:
                base_inst.query_masks = insts[i].query_masks[~is_novel]
            base_insts.append(base_inst)
        
        indices = []
        for i in range(len(base_insts)):
            #is_novel = insts[i].is_novel
            pred_instances = InstanceData_(
                scores=cls_preds[i],
                masks=pred_masks[i])
            gt_instances = InstanceData_(
                labels=base_insts[i].labels_3d,
                masks=base_insts[i].sp_masks)
            if base_insts[i].get('query_masks') is not None:
                gt_instances.query_masks = base_insts[i].query_masks
            indices.append(self.matcher(pred_instances, gt_instances))
        
        ov_insts = []
        ov_pred_masks = []
        mask_maps = []
        for j in range(len(insts)):
            is_novel = insts[j].is_novel
            ov_inst = InstanceData_(
                sp_masks = insts[j].sp_masks[is_novel],
                labels_3d = insts[j].labels_3d[is_novel]
            )
            pred_mask = pred_masks[j].clone()
            mask_ = torch.ones((pred_masks[j].shape[0]), dtype=torch.bool, device=pred_masks[j].device)
            mask_[indices[j][0]] = False
            mask_map = torch.nonzero(mask_)[:, 0]
            mask_maps.append(mask_map)
            if insts[j].get('query_masks') is not None:
                ov_inst.query_masks = insts[j].query_masks[is_novel][:, mask_]
            ov_insts.append(ov_inst)
            ov_pred_mask = pred_mask[mask_]
            ov_pred_masks.append(ov_pred_mask)

        ov_indices = []
        ori_ov_query_indices = []
        for j in range(len(ov_insts)):
            ov_pred_instances = InstanceData_(
                masks=ov_pred_masks[j])
            ov_gt_instances = InstanceData_(
                masks=ov_insts[j].sp_masks,
                labels=ov_insts[j].labels_3d)
            if ov_insts[j].get('query_masks') is not None:
                ov_gt_instances.query_masks = ov_insts[j].query_masks
            ov_indice = self.ov_matcher(ov_pred_instances, ov_gt_instances)
            ori_ov_query_indice = mask_maps[j][ov_indice[0]]
            ori_ov_query_indices.append(ori_ov_query_indice)
            ov_indices.append(ov_indice)
            
        # note:ov has no cls loss    
        cls_losses = []
        for cls_pred, inst, ori_ov_query_indice, (idx_q, idx_gt) in zip(cls_preds, base_insts, ori_ov_query_indices, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_target[ori_ov_query_indice] = -1
            assert torch.isin(idx_q, ori_ov_query_indice).sum() == 0
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight), ignore_index=-1))
        cls_loss = torch.mean(torch.stack(cls_losses))

        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      base_insts, indices):
            if len(idx_q) == 0:
                continue

            pred_mask_base = mask[idx_q]
            tgt_mask_base = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask_base, tgt_mask_base.float()))
            mask_dice_losses.append(dice_loss(pred_mask_base, tgt_mask_base.float()))
            
        score_loss = torch.tensor(0., device=device)

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = torch.tensor(0, device=device)
            mask_dice_loss = torch.tensor(0, device=device)
        
        ov_mask_bce_losses, ov_mask_dice_losses = [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(ov_pred_masks, ov_pred_scores,
                                                      ov_insts, ov_indices):
            if len(idx_q) == 0:
                continue
            pred_mask_ov = mask[idx_q]
            tgt_mask_ov = inst.sp_masks[idx_gt]
            ov_mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask_ov, tgt_mask_ov.float()))
            ov_mask_dice_losses.append(dice_loss(pred_mask_ov, tgt_mask_ov.float()))

        
        if len(ov_mask_bce_losses):
            ov_mask_bce_loss = torch.stack(ov_mask_bce_losses).sum() / len(ov_pred_masks)
            ov_mask_dice_loss = torch.stack(ov_mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                ov_mask_dice_loss = ov_mask_dice_loss / len(ov_pred_masks) * 4
            
            if self.fix_mean_loss:
                ov_mask_bce_loss  = ov_mask_bce_loss * len(ov_pred_masks) \
                    / len(ov_mask_bce_losses)
                ov_mask_dice_loss  = ov_mask_dice_loss * len(ov_pred_masks) \
                    / len(ov_mask_dice_losses)
        else:
            ov_mask_bce_loss = torch.tensor(0., device=device)
            ov_mask_dice_loss = torch.tensor(0., device=device)

        inst_loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss 
            )
        
        ov_inst_loss = (self.loss_weight[4] * ov_mask_bce_loss + 
                        self.loss_weight[5] * ov_mask_dice_loss)
        if self.pred_iou:
            iou_loss = []
            for base_mask_, base_score_, base_inst_, (base_idx_q_, base_idx_gt_), ov_mask_, ov_score_, ov_inst_, (ov_idx_q_, ov_idx_gt_), ori_ov_query_indice_, pred_iou_ in zip(pred_masks, pred_scores,
                                                                                                                                                            base_insts, indices,
                                                                                                                                                            ov_pred_masks, ov_pred_scores,
                                                                                                                                                            ov_insts, ov_indices, ori_ov_query_indices, 
                                                                                                                                                            preds_iou):
                if len(base_idx_q_) == 0 and len(ov_idx_q_) == 0:
                    continue
                if len(base_idx_q_) == 0:
                    ov_pred_mask_ = ov_mask_[ov_idx_q_]
                    ov_tgt_mask_ = ov_inst_.sp_masks[ov_idx_gt_]
                    
                    ori_ov_indice_ = ori_ov_query_indice_
                    
                    total_pred_mask_ = ov_pred_mask_
                    total_tgt_mask_ = ov_tgt_mask_
                    total_indice_ = ori_ov_indice_
                    with torch.no_grad():
                        tgt_score_ = get_iou(total_pred_mask_, total_tgt_mask_).unsqueeze(1)
                    # iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))   
                     
                    filter_id_, _ = torch.where(tgt_score_ >= self.iou_thr)
                    if filter_id_.numel():
                        total_indice_ = total_indice_[filter_id_]
                        tgt_score_ = tgt_score_[filter_id_]
                        iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                        
                elif len(ov_idx_q_) == 0:
                    base_pred_mask_ = base_mask_[base_idx_q_]
                    base_tgt_mask_ = base_inst_.sp_masks[base_idx_gt_]

                    total_pred_mask_ = base_pred_mask_
                    total_tgt_mask_ = base_tgt_mask_
                    total_indice_ = base_idx_q_
                    with torch.no_grad():
                        tgt_score_ = get_iou(total_pred_mask_, total_tgt_mask_).unsqueeze(1)
                    # iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                    
                    filter_id_, _ = torch.where(tgt_score_ >= self.iou_thr)
                    if filter_id_.numel():
                        total_indice_ = total_indice_[filter_id_]
                        tgt_score_ = tgt_score_[filter_id_]
                        iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                    
                else:    
                    base_pred_mask_ = base_mask_[base_idx_q_]
                    base_tgt_mask_ = base_inst_.sp_masks[base_idx_gt_]
                    
                    ov_pred_mask_ = ov_mask_[ov_idx_q_]
                    ov_tgt_mask_ = ov_inst_.sp_masks[ov_idx_gt_]
                    
                    ori_ov_indice_ = ori_ov_query_indice_
                    
                    total_pred_mask_ = torch.cat((base_pred_mask_, ov_pred_mask_), dim=0)
                    total_tgt_mask_ = torch.cat((base_tgt_mask_, ov_tgt_mask_), dim=0)
                    total_indice_ = torch.cat((base_idx_q_, ori_ov_indice_), dim=0)
                    assert torch.isin(ori_ov_indice_ , base_idx_q_).sum() == 0
                    with torch.no_grad():
                        tgt_score_ = get_iou(total_pred_mask_, total_tgt_mask_).unsqueeze(1)
                    # iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                    
                    filter_id_, _ = torch.where(tgt_score_ >= self.iou_thr)
                    if filter_id_.numel():
                        total_indice_ = total_indice_[filter_id_]
                        tgt_score_ = tgt_score_[filter_id_]
                        iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                        
            if len(iou_loss):
                iou_inst_loss = torch.stack(iou_loss).sum() / len(pred_masks) * self.IoU_loss_weight
            else:
                iou_inst_loss = torch.tensor(0., device=device)
                
        if self.pred_iou:
            return inst_loss, ov_inst_loss, iou_inst_loss
        
        return inst_loss, ov_inst_loss
        

    # todo: refactor pred to InstanceData_
    def __call__(self, pred, insts, point_prompt_pred=None, point_prompt_insts=None):
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']
        device = pred_masks[0].device
        #Convenient for subsequent processing, value=[None, ...]
        ov_pred_scores = pred['scores']
        
        if self.pred_iou:
            preds_iou = pred['preds_iou']
        
        base_insts = []
        for i in range(len(insts)):
            is_novel = insts[i].is_novel
            base_inst = InstanceData_(
                labels_3d = insts[i].labels_3d[~is_novel],
                sp_masks = insts[i].sp_masks[~is_novel]
            )
            if insts[i].get('query_masks') is not None:
                base_inst.query_masks = insts[i].query_masks[~is_novel]
            base_insts.append(base_inst)
            
        # match
        indices = []
        for i in range(len(base_insts)):
            #is_novel = insts[i].is_novel
            pred_instances = InstanceData_(
                scores=cls_preds[i],
                masks=pred_masks[i])
            gt_instances = InstanceData_(
                labels=base_insts[i].labels_3d,
                masks=base_insts[i].sp_masks)
            if base_insts[i].get('query_masks') is not None:
                gt_instances.query_masks = base_insts[i].query_masks
            indices.append(self.matcher(pred_instances, gt_instances))

        IntrinsicPointAndQuery_losses = []
        if point_prompt_pred is not None:
            for i, indice in enumerate(indices):
                query_ids, instance_ids = indice
                
                if 'point_prompt_instance_ids' not in point_prompt_insts[i].keys():
                    continue
                if len(instance_ids) < 1:
                    continue

                point_prompt_ids = point_prompt_insts[i].point_prompt_instance_ids
                instance_ids_reverse_mapping = torch.argsort(instance_ids)
                point_prompt_ids_reverse_mapping = torch.argsort(point_prompt_ids)
                if len(instance_ids_reverse_mapping) != len(point_prompt_ids_reverse_mapping):
                    continue
                reverse_mapped_query_ids = query_ids[instance_ids_reverse_mapping]

                query_mask_preds = pred['masks'][i][reverse_mapped_query_ids]
                point_prompt_mask_preds = point_prompt_pred['masks'][i][point_prompt_ids_reverse_mapping]

                max_values, max_indices = torch.max(point_prompt_mask_preds, dim=0)
                num_elements = int(self.ratio * max_values.shape[0])
                if num_elements<1:
                    continue
                point_prompt_mask_preds_slice, Y = torch.topk(max_values, k=num_elements)
                X = max_indices[Y]
                query_mask_preds_slice = query_mask_preds[X,Y]

                query_mask_preds_slice = query_mask_preds_slice.sigmoid()
                point_prompt_mask_preds_slice = point_prompt_mask_preds_slice.sigmoid()
                cur_IntrinsicPointAndQuery_loss = self.bce_criterion(query_mask_preds_slice, point_prompt_mask_preds_slice.detach())

                IntrinsicPointAndQuery_losses.append(cur_IntrinsicPointAndQuery_loss)

        ov_insts = []
        ov_pred_masks = []
        mask_maps = []
        for j in range(len(insts)):
            is_novel = insts[j].is_novel
            ov_inst = InstanceData_(
                sp_masks = insts[j].sp_masks[is_novel],
                labels_3d = insts[j].labels_3d[is_novel]
            )
            pred_mask = pred_masks[j].clone()
            mask_ = torch.ones((pred_masks[j].shape[0]), dtype=torch.bool, device=pred_masks[j].device)
            mask_[indices[j][0]] = False
            mask_map = torch.nonzero(mask_)[:, 0]
            mask_maps.append(mask_map)
            if insts[j].get('query_masks') is not None:
                ov_inst.query_masks = insts[j].query_masks[is_novel][:, mask_]
            ov_insts.append(ov_inst)
            ov_pred_mask = pred_mask[mask_]
            ov_pred_masks.append(ov_pred_mask)

        ov_indices = []
        ori_ov_query_indices = []
        for j in range(len(ov_insts)):
            ov_pred_instances = InstanceData_(
                masks=ov_pred_masks[j])
            ov_gt_instances = InstanceData_(
                masks=ov_insts[j].sp_masks,
                labels=ov_insts[j].labels_3d)
            if ov_insts[j].get('query_masks') is not None:
                ov_gt_instances.query_masks = ov_insts[j].query_masks
            ov_indice = self.ov_matcher(ov_pred_instances, ov_gt_instances)
            ori_ov_query_indice = mask_maps[j][ov_indice[0]]
            ori_ov_query_indices.append(ori_ov_query_indice)
            ov_indices.append(ov_indice)
            
        # class loss, note:ov has no cls loss
        cls_losses = []
        for cls_pred, inst, ori_ov_query_indice, (idx_q, idx_gt) in zip(cls_preds, base_insts, ori_ov_query_indices, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_target[ori_ov_query_indice] = -1
            assert torch.isin(idx_q, ori_ov_query_indice).sum() == 0
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight), ignore_index=-1))
        cls_loss = torch.mean(torch.stack(cls_losses))

        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      base_insts, indices):
            if len(idx_q) == 0:
                continue
            pred_mask_base = mask[idx_q]
            tgt_mask_base = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask_base, tgt_mask_base.float()))
            mask_dice_losses.append(dice_loss(pred_mask_base, tgt_mask_base.float()))

        score_loss = torch.tensor(0, device=device)
        
        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = torch.tensor(0, device=device)
            mask_dice_loss = torch.tensor(0, device=device)

        ov_mask_bce_losses, ov_mask_dice_losses = [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(ov_pred_masks, ov_pred_scores,
                                                      ov_insts, ov_indices):
            if len(idx_q) == 0:
                continue
            pred_mask_ov = mask[idx_q]
            tgt_mask_ov = inst.sp_masks[idx_gt]
            ov_mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask_ov, tgt_mask_ov.float()))
            ov_mask_dice_losses.append(dice_loss(pred_mask_ov, tgt_mask_ov.float()))

        if len(ov_mask_bce_losses):
            ov_mask_bce_loss = torch.stack(ov_mask_bce_losses).sum() / len(ov_pred_masks)
            ov_mask_dice_loss = torch.stack(ov_mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                ov_mask_dice_loss = ov_mask_dice_loss / len(ov_pred_masks) * 4
            
            if self.fix_mean_loss:
                ov_mask_bce_loss  = ov_mask_bce_loss * len(ov_pred_masks) \
                    / len(ov_mask_bce_losses)
                ov_mask_dice_loss  = ov_mask_dice_loss * len(ov_pred_masks) \
                    / len(ov_mask_dice_losses)
        else:
            ov_mask_bce_loss = torch.tensor(0, device=device)
            ov_mask_dice_loss = torch.tensor(0, device=device)
            
        inst_loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss 
            )
        
        ov_inst_loss = (self.loss_weight[4] * ov_mask_bce_loss + 
                        self.loss_weight[5] * ov_mask_dice_loss)
        
        if self.pred_iou:
            iou_loss = []
            for base_mask_, base_score_, base_inst_, (base_idx_q_, base_idx_gt_), ov_mask_, ov_score_, ov_inst_, (ov_idx_q_, ov_idx_gt_), ori_ov_query_indice_, pred_iou_ in zip(pred_masks, pred_scores,
                                                                                                                                                            base_insts, indices,
                                                                                                                                                            ov_pred_masks, ov_pred_scores,
                                                                                                                                                            ov_insts, ov_indices, ori_ov_query_indices, 
                                                                                                                                                            preds_iou):
                if len(base_idx_q_) == 0 and len(ov_idx_q_) == 0:
                    continue
                if len(base_idx_q_) == 0:
                    ov_pred_mask_ = ov_mask_[ov_idx_q_]
                    ov_tgt_mask_ = ov_inst_.sp_masks[ov_idx_gt_]
                    ori_ov_indice_ = ori_ov_query_indice_
                    
                    total_pred_mask_ = ov_pred_mask_
                    total_tgt_mask_ = ov_tgt_mask_
                    total_indice_ = ori_ov_indice_
                    with torch.no_grad():
                        tgt_score_ = get_iou(total_pred_mask_, total_tgt_mask_).unsqueeze(1)
                    #iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                    
                    filter_id_, _ = torch.where(tgt_score_ >= self.iou_thr)
                    if filter_id_.numel():
                        total_indice_ = total_indice_[filter_id_]
                        tgt_score_ = tgt_score_[filter_id_]
                        iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                    
                elif len(ov_idx_q_) == 0:
                    base_pred_mask_ = base_mask_[base_idx_q_]
                    base_tgt_mask_ = base_inst_.sp_masks[base_idx_gt_]

                    total_pred_mask_ = base_pred_mask_
                    total_tgt_mask_ = base_tgt_mask_
                    total_indice_ = base_idx_q_
                    with torch.no_grad():
                        tgt_score_ = get_iou(total_pred_mask_, total_tgt_mask_).unsqueeze(1)
                    #iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                    
                    filter_id_, _ = torch.where(tgt_score_ >= self.iou_thr)
                    if filter_id_.numel():
                        total_indice_ = total_indice_[filter_id_]
                        tgt_score_ = tgt_score_[filter_id_]
                        iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                    
                else:    
                    base_pred_mask_ = base_mask_[base_idx_q_]
                    base_tgt_mask_ = base_inst_.sp_masks[base_idx_gt_]
                    
                    ov_pred_mask_ = ov_mask_[ov_idx_q_]
                    ov_tgt_mask_ = ov_inst_.sp_masks[ov_idx_gt_]
                    
                    ori_ov_indice_ = ori_ov_query_indice_
                    
                    total_pred_mask_ = torch.cat((base_pred_mask_, ov_pred_mask_), dim=0)
                    total_tgt_mask_ = torch.cat((base_tgt_mask_, ov_tgt_mask_), dim=0)
                    total_indice_ = torch.cat((base_idx_q_, ori_ov_indice_), dim=0)
                    assert torch.isin(ori_ov_indice_ , base_idx_q_).sum() == 0
                    with torch.no_grad():
                        tgt_score_ = get_iou(total_pred_mask_, total_tgt_mask_).unsqueeze(1)
                    #iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
                    
                    filter_id_, _ = torch.where(tgt_score_ >= self.iou_thr)
                    if filter_id_.numel():
                        total_indice_ = total_indice_[filter_id_]
                        tgt_score_ = tgt_score_[filter_id_]
                        iou_loss.append(F.mse_loss(pred_iou_[total_indice_], tgt_score_))
            if len(iou_loss):
                iou_inst_loss = torch.stack(iou_loss).sum() / len(pred_masks) * self.IoU_loss_weight
            else:
                iou_inst_loss = torch.tensor(0., device=device)   
                                     
        if self.pred_iou:
            if 'aux_outputs' in pred:
                if self.iter_matcher:
                    indices = None
                for i, aux_outputs in enumerate(pred['aux_outputs']):
                    loss = self.get_layer_loss(aux_outputs, insts, indices)
                    inst_loss += loss[0]
                    ov_inst_loss += loss[1]
                    iou_inst_loss += loss[2]
                    
            return_loss = {'inst_loss': inst_loss, 'ov_inst_loss':ov_inst_loss, 'iou_inst_loss':iou_inst_loss}

            if point_prompt_pred is not None and len(IntrinsicPointAndQuery_losses)>0:
                return_loss.update({'IntrinsicPointAndQuery_loss':self.IntrinsicPointAndQuery_loss_weight*torch.mean(torch.stack(IntrinsicPointAndQuery_losses))})
            return return_loss
        
        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss = self.get_layer_loss(aux_outputs, insts, indices)
                inst_loss += loss[0]
                ov_inst_loss += loss[1]

        return_loss = {'inst_loss': inst_loss, 'ov_inst_loss':ov_inst_loss}
        if point_prompt_pred is not None and len(IntrinsicPointAndQuery_losses)>0:
            return_loss.update({'IntrinsicPointAndQuery_loss':self.IntrinsicPointAndQuery_loss_weight*torch.mean(torch.stack(IntrinsicPointAndQuery_losses))})
        return return_loss


@TASK_UTILS.register_module()
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `scores` of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        scores = pred_instances.scores.softmax(-1)
        cost = -scores[:, gt_instances.labels]
        return cost * self.weight


@TASK_UTILS.register_module()
class MaskBCECost:
    """Sigmoid BCE cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_sigmoid_bce_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight


@TASK_UTILS.register_module()
class MaskDiceCost:
    """Dice cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_dice_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight

@TASK_UTILS.register_module()
class SparseMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """

    def __init__(self, costs, topk):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points),
                `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(
            gt_instances.query_masks.T, cost_value, self.inf)

        values = torch.topk(
            cost_value, self.topk + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]
