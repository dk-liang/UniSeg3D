from mmdet3d.registry import MODELS
from .structures import InstanceData_
from torch import nn


@MODELS.register_module()
class ScanNetUnifiedCriterion(nn.Module):
    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion,
                 point_prompt_inst_criterion, text_prompt_inst_criterion, contrastive_criterion):
        super(ScanNetUnifiedCriterion, self).__init__()
        self.num_semantic_classes = num_semantic_classes
        self.sem_criterion = MODELS.build(sem_criterion)
        self.inst_criterion = MODELS.build(inst_criterion)
        self.point_prompt_inst_criterion = MODELS.build(point_prompt_inst_criterion)
        self.text_prompt_inst_criterion = MODELS.build(text_prompt_inst_criterion)
        self.contrastive_criterion = MODELS.build(contrastive_criterion)
    
    def __call__(self, pred, insts,
                 point_prompt_pred=None, point_prompt_insts=None,
                 text_prompt_pred=None, text_prompt_insts=None):
        sem_gts = []
        inst_gts = []
        point_prompt_gts = []
        text_prompt_gts = []
        n = self.num_semantic_classes

        for i in range(len(pred['masks'])):
            sem_gt = InstanceData_()
            if insts[i].get('query_masks') is not None:
                sem_gt.sp_masks = insts[i].query_masks[-n - 1:, :]
            else:
                sem_gt.sp_masks = insts[i].sp_masks[-n - 1:, :]
            sem_gts.append(sem_gt)

            inst_gt = InstanceData_()
            inst_gt.sp_masks = insts[i].sp_masks[:-n - 1, :]
            inst_gt.labels_3d = insts[i].labels_3d[:-n - 1]
            inst_gt.is_novel = insts[i].is_novel[:-n - 1]
            if insts[i].get('query_masks') is not None:
                inst_gt.query_masks = insts[i].query_masks[:-n - 1, :]
            inst_gts.append(inst_gt)

            if point_prompt_pred is not None:
                point_prompt_gt = InstanceData_()
                point_prompt_gt.sp_masks = point_prompt_insts[i].sp_masks
                point_prompt_gt.labels_3d = point_prompt_insts[i].labels_3d
                point_prompt_gt.point_object_id = point_prompt_insts[i].point_object_id
                if 'point_prompt_distance_norms' in point_prompt_insts[i].keys():
                    point_prompt_gt.point_prompt_distance_norms = point_prompt_insts[i].point_prompt_distance_norms
                if point_prompt_insts[i].get('query_masks') is not None:
                    point_prompt_gt.query_masks = point_prompt_insts[i].query_masks

                point_prompt_gts.append(point_prompt_gt)

            if text_prompt_pred is not None:
                text_prompt_gt = InstanceData_()
                text_prompt_gt.sp_masks = text_prompt_insts[i].sp_masks
                text_prompt_gt.labels_3d = text_prompt_insts[i].labels_3d
                text_prompt_gt.text_object_id = text_prompt_insts[i].text_object_id
                if text_prompt_insts[i].get('query_masks') is not None:
                    text_prompt_gt.query_masks = text_prompt_insts[i].query_masks

                text_prompt_gts.append(text_prompt_gt)
                
        loss = self.inst_criterion(pred, inst_gts, point_prompt_pred=point_prompt_pred, point_prompt_insts=point_prompt_insts)
        loss.update(self.sem_criterion(pred, sem_gts))
        
        if point_prompt_pred is not None:
            loss.update(self.point_prompt_inst_criterion(point_prompt_pred, point_prompt_gts))

        if text_prompt_pred is not None:
            loss.update(self.text_prompt_inst_criterion(text_prompt_pred, text_prompt_gts))

        loss.update(self.contrastive_criterion(pred, inst_gts, point_prompt_pred, point_prompt_gts, text_prompt_pred, text_prompt_gts))
        return loss