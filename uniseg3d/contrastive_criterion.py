import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import random
import numpy as np
from .structures import InstanceData_
from mmdet3d.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class ContrastiveCriterion(nn.Module):
    def __init__(self, use_ranking, contrastive_loss_weight, ranking_loss_weight,
                 cls_logit_loss_weight=None, use_pseudo_cls_supervise=False):
        super(ContrastiveCriterion, self).__init__()
        self.use_ranking = use_ranking
        self.contrastive_loss_weight = contrastive_loss_weight
        self.ranking_loss_weight = ranking_loss_weight
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if self.use_ranking:
            self.rank_loss = nn.MarginRankingLoss()
        self.use_pseudo_cls_supervise = use_pseudo_cls_supervise
        if self.use_pseudo_cls_supervise:
            self.cls_logit_loss_weight = cls_logit_loss_weight
            self.cls_logit_criterion = nn.BCELoss()

    def compute_losses(self, logits, gather_type='train'):
        losses = {}

        z = torch.zeros(1).cuda()
        losses["rank_loss"] = z
        if gather_type == 'train':
            y1 = torch.ones(1).cuda()
            for i in range(logits.size()[0]):
                for j in range(logits.size()[0]):
                    if i != j:
                        losses["rank_loss"] += self.rank_loss(logits[j][j].unsqueeze(0), logits[i][j].unsqueeze(0), y1)
        return losses

    def __call__(self, pred, inst_gts, point_prompt_pred=None, point_prompt_gts=None, text_prompt_pred=None, text_prompt_gts=None):

        one = torch.tensor(1.).to(pred['masks'][0].device)
        zero = torch.tensor(0.).to(pred['masks'][0].device)

        losses = []
        ranking_losses = []
        cls_logit_losses = []
        for i in range(len(pred['masks'])):
            
            emb_point = point_prompt_pred['contras_embeds'][i]
            emb_text = text_prompt_pred['contras_embeds'][i]
            text_prompt_gt = text_prompt_gts[i]
            point_prompt_gt = point_prompt_gts[i]
            # query_gt = inst_gts[i]

            '''text vs point'''
            if len(text_prompt_gt.text_object_id) == 0:
            
                continue

            use_point_prompt_distance_norm = 'point_prompt_distance_norms' in point_prompt_gt.keys()
            refer_embed = []
            if use_point_prompt_distance_norm:
                point_prompt_distance_norms = []
            if self.use_pseudo_cls_supervise:
                point_prompt_cls_preds = []
            for _, id_text in enumerate(text_prompt_gt.text_object_id):
                pos_ind = point_prompt_gt.point_object_id == id_text
                refer_embed.append(emb_point[pos_ind])
                if use_point_prompt_distance_norm:
                    point_prompt_distance_norms.append(point_prompt_gt['point_prompt_distance_norms'][pos_ind])
                if self.use_pseudo_cls_supervise:
                    point_prompt_cls_preds.append(point_prompt_pred['cls_preds'][i][pos_ind])

            refer_embed = torch.cat(refer_embed, dim=0)
            if use_point_prompt_distance_norm:
                point_prompt_distance_norms = torch.cat(point_prompt_distance_norms, dim=0)
            if self.use_pseudo_cls_supervise:
                point_prompt_cls_preds = torch.cat(point_prompt_cls_preds, dim=0)
            key_embed = emb_text
            refer_embed = F.normalize(refer_embed, dim=-1)
            key_embed = F.normalize(key_embed, dim=-1)

            logits_point = self.logit_scale * refer_embed @ key_embed.T
            logits_text = self.logit_scale * key_embed @ refer_embed.T
            labels = self.get_ground_truth(zero.device, logits_point.shape[0])
            loss = (
                                 F.cross_entropy(logits_point, labels, reduction='none') +  # point prompt 的损失
                                 F.cross_entropy(logits_text, labels, reduction='none')  # text prompt 的损失
                         ) / 2
            if use_point_prompt_distance_norm:
                loss = loss*point_prompt_distance_norms
            if len(loss) > 0:
                loss = torch.mean(loss)
            losses.append(loss)
            
            if self.use_ranking:
                logits = refer_embed@key_embed.T
                rank_loss = self.compute_losses(logits)
                ranking_losses.append(rank_loss["rank_loss"])

            if self.use_pseudo_cls_supervise:
                point_cls_logits = point_prompt_cls_preds.sigmoid()
                text_cls_logits = text_prompt_pred['cls_preds'][i].sigmoid()
                cls_logit_losses.append(self.cls_logit_criterion(text_cls_logits, point_cls_logits.detach()))

        if len(losses) == 0:
            output = {'contrastive_loss': zero}
        else:
            output = {'contrastive_loss': self.contrastive_loss_weight*torch.mean(torch.stack(losses))}
            if self.use_ranking and len(ranking_losses)>0:
                output.update({'rank_loss': self.ranking_loss_weight*torch.mean(torch.stack(ranking_losses))})
            if self.use_pseudo_cls_supervise and len(cls_logit_losses)>0:
                output.update({'pseudo_ditribute_loss': self.cls_logit_loss_weight*torch.mean(torch.stack(cls_logit_losses))})
        return output            
        
    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        return labels