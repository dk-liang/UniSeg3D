import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .structures import InstanceData_
from mmdet3d.registry import MODELS, TASK_UTILS


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
class TextPromptInstanceCriterion:
    def __init__(self, loss_weight, non_object_weight, num_classes,
                 fix_dice_loss_weight, fix_mean_loss=False, total_weight=1.0,
                 use_cls_supervise=False):
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.fix_mean_loss = fix_mean_loss
        self.total_weight = total_weight
        if use_cls_supervise:
            self.cls_weight = 1.
        else:
            self.cls_weight = 0.

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks)
                if insts[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts[i].query_masks
                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            if cls_pred is None: continue
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        if not cls_losses: return 0
        cls_loss = torch.mean(torch.stack(cls_losses))

        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if mask is None: continue
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

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
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss * self.cls_weight +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        return loss

    # todo: refactor pred to InstanceData_
    def __call__(self, pred, insts):
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # match
        indices = []
        for i in range(len(insts)):
            query_index=torch.tensor([m for m in range(len(insts[i].labels_3d))]).to(insts[i].labels_3d.device)
            labels_index=query_index.clone()
            indices.append((query_index,labels_index))

        # class loss
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            if cls_pred is None: continue
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        if not cls_losses: return {'text_prompt_inst_loss': torch.tensor(0.).to(insts[0].sp_masks.device)}
        cls_loss = torch.mean(torch.stack(cls_losses))

        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if mask is None: continue
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        
        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss * self.cls_weight +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        return {'text_prompt_inst_loss': self.total_weight * loss}
