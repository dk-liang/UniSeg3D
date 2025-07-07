import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean
import MinkowskiEngine as ME

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from .mask_matrix_nms import mask_matrix_nms

from mmdet3d.registry import TASK_UTILS
from .structures import InstanceData_

import copy


class UniSeg3DMixin:
    def point_prompt_predict_by_feat(self, out, superpoints, point_prompt_instances):
        if out['masks'][0] == None:
            obj_gt = torch.as_tensor([])
            obj_res = torch.as_tensor([])
        else:
            obj_res = self.point_prompt_predict_by_feat_object(out, superpoints)
            obj_gt = point_prompt_instances[0].sp_masks[:,superpoints]

        return [
            PointData(
                point_prompt_object_pred_masks=obj_res.cpu().numpy(),
                point_prompt_object_gt_masks=obj_gt.cpu().numpy())]

    def point_prompt_predict_by_feat_object(self, out, superpoints):
        mask_pred = out['masks'][0]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        
        return mask_pred
    
    def text_prompt_predict_by_feat(self, out, superpoints, text_prompt_instances):
        if out['masks'][0] == None:
            obj_gt = torch.as_tensor([])
            obj_res = torch.as_tensor([])
        else:
            obj_res = self.text_prompt_predict_by_feat_object(out, superpoints)
            obj_gt = text_prompt_instances[0].sp_masks[:,superpoints]

        return [
            PointData(
                text_prompt_object_pred_masks=obj_res.cpu().numpy(),
                text_prompt_object_gt_masks=obj_gt.cpu().numpy())]

    def text_prompt_predict_by_feat_object(self, out, superpoints):
        mask_pred = out['masks'][0]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        
        return mask_pred
          
    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.inst_score_thr, is_test_iou=self.inst_test_iou)
        sem_res = self.predict_by_feat_semantic(out, superpoints)
        pan_res = self.predict_by_feat_panoptic(out, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold, is_test_iou=False):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if is_test_iou:
            if out['preds_iou'][0] is not None:
                scores *= out['preds_iou'][0]
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
    
        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def predict_by_feat_semantic(self, out, superpoints, classes=None):
        """Predict semantic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `sem_preds` of shape (n_queries, n_semantic_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            classes (List[int] or None): semantic (stuff) class ids.
        
        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, n_semantic_classe + 1),
        """
        if classes is None:
            classes = list(range(out['sem_preds'][0].shape[1] - 1))
        return out['sem_preds'][0][:, classes].argmax(dim=1)[superpoints]

    def predict_by_feat_panoptic(self, out, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        sem_map = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        mask_pred, labels, scores  = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr, is_test_iou=self.pano_test_iou)
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask
        return sem_map, inst_map
    
    def _select_queries(self, x, gt_instances):
        """Select queries for train pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, n_channels).
            gt_instances (List[InstanceData_]): of len batch_size.
                Ground truth which can contain `labels` of shape (n_gts_i,),
                `sp_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                List[InstanceData_]: of len batch_size, each updated
                    with `query_masks` of shape (n_gts_i, n_queries_i).
        """
        queries = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                n = torch.as_tensor(min(n, 3500)).item()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
            else:
                queries.append(x[i])
                gt_instances[i].query_masks = gt_instances[i].sp_masks
            
        return queries, gt_instances


@MODELS.register_module()
class UniSeg3D(UniSeg3DMixin, Base3DDetector):
    r"""UniSeg3D for ScanNet dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 query_thr,
                 set_query_mask=True,
                 set_all_mask=False,
                 is_type_embedding=False,
                 alpha = 0.4,
                 beta = 0.8,
                 pred_iou = False,
                 inst_test_iou = False,
                 pano_test_iou = True,
                 overlap = None,
                 backbone=None,
                 decoder=None,
                 lang=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.lang = MODELS.build(lang)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
        
        self.set_query_mask = set_query_mask
        self.set_all_mask = set_all_mask
        self.is_type_embedding = is_type_embedding
        if self.is_type_embedding:
            self.type_embedding = torch.nn.Embedding(3, num_channels)
        
        self.alpha = alpha
        self.beta = beta

        self.pred_iou = pred_iou
        d_model = decoder['d_model']
        if self.pred_iou:
            self.pred_iou_proj = nn.Sequential(
                    nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                    nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                    nn.Linear(d_model, 1))

        self.inst_test_iou = inst_test_iou
        self.pano_test_iou = pano_test_iou

        self.overlap = overlap
        
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i]: batch_offsets[i + 1]])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        sp_pts_masks = []
        point_prompt_outputs = None
        point_prompt_instances = None
        text_prompt_outputs = None
        text_prompt_instances = None
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg

            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        queries, sp_gt_instances = self._select_queries(x, sp_gt_instances)
        
        if self.set_query_mask:
            original_lengths = []
            for query in queries:
                original_lengths.append(query.shape[0])
                
        if 'point_prompts' in batch_inputs_dict.keys():
            queries, point_prompt_instances = self.add_point_prompt_queries(queries, sp_gt_instances, batch_inputs_dict, x, batch_offsets)
        if 'text_token' in batch_inputs_dict.keys():
            queries, text_prompt_instances = self.add_text_prompt_queries(queries, sp_gt_instances, batch_inputs_dict, x, batch_offsets)
        
        if self.set_query_mask:
            interaction_masks = []
            for m, query in enumerate(queries):
                cur_interaction_mask = torch.zeros((query.shape[0], query.shape[0])).bool().to(query.device)
                cur_interaction_mask[:original_lengths[m], original_lengths[m]: ] = True
                if self.set_all_mask:
                    cur_interaction_mask[original_lengths[m]:, :original_lengths[m]] = True
                    if query.shape[0] - original_lengths[m] > 0:
                        size = query.shape[0] - original_lengths[m]
                        inter_mask = torch.eye(size).bool().to(query.device)
                        cur_interaction_mask[original_lengths[m]:, original_lengths[m]:] = ~inter_mask
                interaction_masks.append(cur_interaction_mask)
                pass
        else:
            interaction_masks = None    
            
        x = self.decoder(x, queries, interaction_masks=interaction_masks)
        
        if 'text_token' in batch_inputs_dict.keys():
            x, text_prompt_outputs = self.split_original_and_text_prompt_queries(x, batch_inputs_dict)
        if 'point_prompts' in batch_inputs_dict.keys():
            x, point_prompt_outputs = self.split_original_and_point_prompt_queries(x, batch_inputs_dict)

        if self.pred_iou:
            x = self.get_pred_iou(x)

        loss = self.criterion(x, sp_gt_instances, point_prompt_outputs, point_prompt_instances, text_prompt_outputs, text_prompt_instances)
        return loss

    def get_pred_iou(self, x):
        batch = len(x['contras_embeds'])
        num_layers = len(x['aux_outputs'])
        preds_iou = []
        for i in range(batch):
            contras_embed = x['contras_embeds'][i].clone()
            pred = self.pred_iou_proj(contras_embed)
            preds_iou.append(pred)
        x['preds_iou'] = preds_iou
        
        for i in range(num_layers):
            preds_iou_aux = []
            for j in range(batch):
                contras_embed_aux = x['aux_outputs'][i]['contras_embeds'][j].clone()
                pred_aux = self.pred_iou_proj(contras_embed_aux)
                preds_iou_aux.append(pred_aux)
            x['aux_outputs'][i]['preds_iou'] = preds_iou_aux
        pass
    
        return x

    def split_original_and_text_prompt_queries(self, x, batch_inputs_dict, mode='train'):
        batch_label_text = batch_inputs_dict['label_text']
        batch_gt_text_prompts = batch_inputs_dict['gt_text_prompt']

        batch_size = len(batch_gt_text_prompts)
        num_layers = len(x['aux_outputs'])
        text_prompt_outputs = {
            'cls_preds':[None for b in range(batch_size)],
            'sem_preds':[None for b in range(batch_size)],
            'masks':[None for b in range(batch_size)],
            'scores':[None for b in range(batch_size)],
            'contras_embeds':[None for b in range(batch_size)],
        }
        original_outputs = {
            'cls_preds':[None for b in range(batch_size)],
            'sem_preds':[None for b in range(batch_size)],
            'masks':[None for b in range(batch_size)],
            'scores':[None for b in range(batch_size)],
            'contras_embeds':[None for b in range(batch_size)],
        }
        keys = list(text_prompt_outputs.keys())
        text_prompt_outputs.update({
            'aux_outputs': [{key:[None for b in range(batch_size)] for key in keys} for b in range(num_layers)]
        })
        original_outputs.update({
            'aux_outputs': [{key:[None for b in range(batch_size)] for key in keys} for b in range(num_layers)]
        })

        for key in keys:
            for i, cur_text_prompts in enumerate(batch_gt_text_prompts):
                if x[key][i] is None: continue
                cur_num_text_prompts = len(cur_text_prompts)
                if cur_num_text_prompts > 0:
                    if mode == 'train':
                        text_prompt_outputs[key][i]=x[key][i][-cur_num_text_prompts:]
                        original_outputs[key][i]=x[key][i][:-cur_num_text_prompts]
                    elif mode == 'val':
                        if key == 'masks':
                            text_prompt_outputs[key][i]=x[key][i][-cur_num_text_prompts:, :]
                            original_outputs[key][i]=x[key][i][:-cur_num_text_prompts, :]
                        else:
                            text_prompt_outputs[key][i]=x[key][i][-cur_num_text_prompts:]
                            original_outputs[key][i]=x[key][i][:-cur_num_text_prompts]
                else:
                    original_outputs[key][i]=x[key][i]

                for aux_layer in range(num_layers):
                    if x['aux_outputs'][aux_layer][key] is None:
                        text_prompt_outputs['aux_outputs'][aux_layer][key]=None
                        continue
                    if x['aux_outputs'][aux_layer][key][i] is None: continue
                    if cur_num_text_prompts > 0:
                        if mode == 'train':
                            text_prompt_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][-cur_num_text_prompts:]
                            original_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][:-cur_num_text_prompts]
                        elif mode == 'val':
                            if key == 'masks':
                                text_prompt_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][-cur_num_text_prompts:, :]
                                original_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][:-cur_num_text_prompts, :]
                            else:
                                text_prompt_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][-cur_num_text_prompts:]
                                original_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][:-cur_num_text_prompts]   
                    else:
                        original_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i]

        return original_outputs, text_prompt_outputs

    def split_original_and_point_prompt_queries(self, x, batch_inputs_dict, mode='train'):
        batch_point_prompts = batch_inputs_dict['point_prompts']
        batch_point_prompt_instance_ids = batch_inputs_dict['point_prompt_instance_ids']
        batch_point_prompt_sp_ids = batch_inputs_dict['point_prompt_sp_ids']

        batch_size = len(batch_point_prompts)
        num_layers = len(x['aux_outputs'])
        point_prompt_outputs = {
            'cls_preds':[None for b in range(batch_size)],
            'sem_preds':[None for b in range(batch_size)],
            'masks':[None for b in range(batch_size)],
            'scores':[None for b in range(batch_size)],
            'contras_embeds':[None for b in range(batch_size)],
        }
        original_outputs = {
            'cls_preds':[None for b in range(batch_size)],
            'sem_preds':[None for b in range(batch_size)],
            'masks':[None for b in range(batch_size)],
            'scores':[None for b in range(batch_size)],
            'contras_embeds':[None for b in range(batch_size)],
        }
        keys = list(point_prompt_outputs.keys())
        point_prompt_outputs.update({
            'aux_outputs': [{key:[None for b in range(batch_size)] for key in keys} for b in range(num_layers)]
        })
        original_outputs.update({
            'aux_outputs': [{key:[None for b in range(batch_size)] for key in keys} for b in range(num_layers)]
        })

        for key in keys:
            for i, cur_point_prompts in enumerate(batch_point_prompts):
                if x[key][i] is None: continue
                cur_num_point_prompts = len(cur_point_prompts)
                if cur_num_point_prompts > 0:
                    if mode == 'train':
                        point_prompt_outputs[key][i]=x[key][i][-cur_num_point_prompts:]
                        original_outputs[key][i]=x[key][i][:-cur_num_point_prompts]
                    elif mode == 'val':
                        if key == 'masks':
                            point_prompt_outputs[key][i]=x[key][i][-cur_num_point_prompts:, :]
                            original_outputs[key][i]=x[key][i][:-cur_num_point_prompts, :]
                        else:
                            point_prompt_outputs[key][i]=x[key][i][-cur_num_point_prompts:]
                            original_outputs[key][i]=x[key][i][:-cur_num_point_prompts]
                else:
                    original_outputs[key][i]=x[key][i]

                for aux_layer in range(num_layers):
                    if x['aux_outputs'][aux_layer][key] is None:
                        point_prompt_outputs['aux_outputs'][aux_layer][key]=None
                        continue
                    if x['aux_outputs'][aux_layer][key][i] is None: continue
                    if cur_num_point_prompts > 0:
                        if mode == 'train':
                            point_prompt_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][-cur_num_point_prompts:]
                            original_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][:-cur_num_point_prompts]
                        elif mode == 'val':
                            if key == 'masks':
                                point_prompt_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][-cur_num_point_prompts:, :]
                                original_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][:-cur_num_point_prompts, :]
                            else:
                                point_prompt_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][-cur_num_point_prompts:]
                                original_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i][:-cur_num_point_prompts]   
                    else:
                        original_outputs['aux_outputs'][aux_layer][key][i]=x['aux_outputs'][aux_layer][key][i]

        return original_outputs, point_prompt_outputs

    def add_text_prompt_queries(self, queries, sp_gt_instances, batch_inputs_dict, batch_sp_feat, batch_offsets):
        batch_text_prompts = self.lang(batch_inputs_dict['text_token'])
        batch_label_text = batch_inputs_dict['label_text']
        batch_gt_text_prompts = batch_inputs_dict['gt_text_prompt']

        device = batch_inputs_dict['points'][0].device
        queries = [query.clone() for query in queries]
        text_prompt_gt_instances = [InstanceData_() for _ in range(len(sp_gt_instances))]
        
        for i, cur_text_prompts in enumerate(batch_text_prompts):
            if len(cur_text_prompts) > 0:
                cur_text_prompts = cur_text_prompts + self.type_embedding.weight[0]
                    
                queries[i] = torch.cat((queries[i],cur_text_prompts), 0)
                # instance gt
                text_prompt_gt_instances[i].labels_3d = batch_label_text[i].clone().long()
                # sp_masks
                text_prompt_gt_instances[i].sp_masks = batch_gt_text_prompts[i].clone().bool()
                # object_id
                text_prompt_gt_instances[i].text_object_id = batch_inputs_dict['text_object_id'][i]
            else:
                # instance gt
                text_prompt_gt_instances[i].labels_3d = torch.tensor([]).to(device)
                # sp_masks
                text_prompt_gt_instances[i].sp_masks = torch.tensor([]).to(device)
                # object_id
                text_prompt_gt_instances[i].text_object_id = torch.tensor([]).to(device)

        return queries, text_prompt_gt_instances

    def add_point_prompt_queries(self, queries, sp_gt_instances, batch_inputs_dict, batch_sp_feat, batch_offsets):
        batch_point_prompts = batch_inputs_dict['point_prompts']
        batch_point_prompt_instance_ids = batch_inputs_dict['point_prompt_instance_ids']
        batch_point_prompt_sp_ids = batch_inputs_dict['point_prompt_sp_ids']
        queries = [query.clone() for query in queries]
        if self.is_type_embedding:
            queries = [query + self.type_embedding.weight[2].unsqueeze(0) for query in queries]
        
            
        point_prompt_gt_instances = [InstanceData_() for _ in range(len(sp_gt_instances))]

        for i, cur_point_prompts in enumerate(batch_point_prompts):
            cur_point_prompt_queries_list = []
            cur_point_prompt_query_masks_list = []
            cur_point_prompt_labels_3d_list = []
            cur_point_prompt_sp_masks_list = []

            for j in range(len(cur_point_prompts)):
                cur_point_prompt_sp_ids = batch_point_prompt_sp_ids[i][j]
                cur_point_prompt_instance_ids = batch_point_prompt_instance_ids[i][j]

                cur_point_prompt_feat = batch_sp_feat[i][cur_point_prompt_sp_ids:cur_point_prompt_sp_ids+1]
                if self.is_type_embedding:
                    cur_point_prompt_feat = cur_point_prompt_feat + self.type_embedding.weight[1]
                cur_point_prompt_queries_list.append(cur_point_prompt_feat)
                assert len(sp_gt_instances[i]) > 0
                # instance gt
                cur_point_prompt_label_3d = sp_gt_instances[i].labels_3d[cur_point_prompt_instance_ids:cur_point_prompt_instance_ids+1].clone()
                cur_point_prompt_labels_3d_list.append(cur_point_prompt_label_3d)
                # sp_masks
                cur_point_prompt_sp_mask = sp_gt_instances[i].sp_masks[cur_point_prompt_instance_ids:cur_point_prompt_instance_ids+1].clone()
                cur_point_prompt_sp_masks_list.append(cur_point_prompt_sp_mask)

            if cur_point_prompt_queries_list:
                cur_point_prompt_queries = torch.cat(cur_point_prompt_queries_list, dim=0)
                    
                queries[i] = torch.cat((queries[i], cur_point_prompt_queries), 0)
            
                # instance gt
                cur_point_prompt_labels_3d = torch.cat(cur_point_prompt_labels_3d_list)
                point_prompt_gt_instances[i].labels_3d = cur_point_prompt_labels_3d
                # sp_masks
                cur_point_prompt_sp_masks = torch.cat(cur_point_prompt_sp_masks_list)
                point_prompt_gt_instances[i].sp_masks = cur_point_prompt_sp_masks
                # object_id_shuffle
                point_prompt_gt_instances[i].point_object_id = batch_inputs_dict['pts_instance_objextId_shuffle'][i]
                # point_prompt_distance_norms
                if 'point_prompt_distance_norms' in batch_inputs_dict.keys():
                    point_prompt_gt_instances[i].point_prompt_distance_norms = batch_inputs_dict['point_prompt_distance_norms'][i]
                if 'point_prompt_instance_ids' in batch_inputs_dict.keys():
                    point_prompt_gt_instances[i].point_prompt_instance_ids = batch_inputs_dict['point_prompt_instance_ids'][i]
            else:
                point_prompt_gt_instances[i].labels_3d = torch.tensor([])
                point_prompt_gt_instances[i].sp_masks = torch.tensor([])
                point_prompt_gt_instances[i].point_object_id = torch.tensor([])
                point_prompt_gt_instances[i].point_prompt_distance_norms = torch.tensor([])
                
                

        return queries, point_prompt_gt_instances

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        
        queries = copy.deepcopy(x)
        
        if self.set_query_mask:
            original_lengths = []
            for query in queries:
                original_lengths.append(query.shape[0])
                
        if 'point_prompts' in batch_inputs_dict.keys():
            queries, point_prompt_instances = self.add_point_prompt_queries(queries, sp_gt_instances, batch_inputs_dict, x, batch_offsets)
        if 'text_token' in batch_inputs_dict.keys():
            queries, text_prompt_instances = self.add_text_prompt_queries(queries, sp_gt_instances, batch_inputs_dict, x, batch_offsets)

        if self.set_query_mask:
            interaction_masks = []
            for m, query in enumerate(queries):
                cur_interaction_mask = torch.zeros((query.shape[0], query.shape[0])).bool().to(query.device)
                cur_interaction_mask[:original_lengths[m], original_lengths[m]: ] = True
                if self.set_all_mask:
                    cur_interaction_mask[original_lengths[m]:, :original_lengths[m]] = True
                    if query.shape[0] - original_lengths[m] > 0:
                        size = query.shape[0] - original_lengths[m]
                        inter_mask = torch.eye(size).bool().to(query.device)
                        cur_interaction_mask[original_lengths[m]:, original_lengths[m]:] = ~inter_mask
                interaction_masks.append(cur_interaction_mask)
                pass
        else:
            interaction_masks = None
            
        x = self.decoder(x, queries, interaction_masks=interaction_masks)
        
        if 'text_token' in batch_inputs_dict.keys():
            x, text_prompt_outputs = self.split_original_and_text_prompt_queries(x, batch_inputs_dict, mode='val') 
            text_prompt_results_list = self.text_prompt_predict_by_feat(text_prompt_outputs, sp_pts_masks, text_prompt_instances)
        if 'point_prompts' in batch_inputs_dict.keys():
            x, point_prompt_outputs = self.split_original_and_point_prompt_queries(x, batch_inputs_dict, mode='val')
            point_prompt_results_list = self.point_prompt_predict_by_feat(point_prompt_outputs, sp_pts_masks, point_prompt_instances)
        
        if  self.pred_iou:
            x = self.get_pred_iou(x)

            ##############################################################################################################
            '''
            Open-vocabulary inference: Uncomment this entire block to generate class-agnostic masks


            Instructions:
            1. Replace 'your_save_folder_path_here' with your desired output directory
            2. Adjust iou_threshold and mask_threshold if needed
            3. Uncomment the entire block below
            '''

            # import os

            # # Configuration - Modify these values as needed
            # save_path = 'your_save_folder_path_here'  # TODO: Replace with your save folder path
            # iou_threshold = 0.6                       # IoU threshold for mask filtering
            # mask_threshold = 0.4                      # Mask binarization threshold

            # # Create output directory
            # os.makedirs(save_path, exist_ok=True)

            # # Generate class-agnostic masks
            # preds_iou_ = x['preds_iou'][0].clone()
            # pred_masks_ = x['masks'][0].clone()
            # pred_masks_sigmoid_ = pred_masks_.sigmoid()

            # # Calculate mask scores
            # mask_score_ = (pred_masks_sigmoid_ * (pred_masks_ > 0)).sum(1) / ((pred_masks_ > 0).sum(1) + 1e-6)
            # mask_iou_score_ = mask_score_ * preds_iou_.squeeze(-1)

            # # Filter masks by IoU threshold
            # idx = mask_iou_score_ > iou_threshold

            # # Prepare save path
            # scene_path = os.path.join(save_path, 
            #                         batch_data_samples[0].lidar_path.split('/')[-1].replace('bin', 'pth'))

            # # Save results
            # info = dict()
            # info['ins'] = (pred_masks_sigmoid_[idx][:, sp_pts_masks] > mask_threshold).cpu().numpy().astype('uint8')
            # info['conf'] = mask_iou_score_[idx].cpu().numpy()
            # torch.save(info, scene_path)

            # print(f"Class-agnostic masks saved to: {scene_path}")
            ##############################################################################################################

        results_list = self.predict_by_feat(x, sp_pts_masks)
            
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
            if 'point_prompts' in batch_inputs_dict.keys():
                data_sample.point_prompt_pred_pts_seg = point_prompt_results_list[i]

                prob_ = F.softmax(point_prompt_outputs['cls_preds'][i], dim=-1)[:, :-1]
                point_prob = torch.gather(prob_, 1, point_prompt_instances[0].labels_3d.unsqueeze(-1)).squeeze(-1)
                data_sample.point_prob = point_prob

            if 'text_token' in batch_inputs_dict.keys():
                data_sample.text_prompt_pred_pts_seg = text_prompt_results_list[i]
        return batch_data_samples
    