import numpy as np
import scipy
import torch
import random
from torch_scatter import scatter_mean
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import PointSample
from mmdet3d.registry import TRANSFORMS

import MinkowskiEngine as ME


@TRANSFORMS.register_module()
class TextPromptTest(BaseTransform):
    def __init__(self, num_ins = 3, random_select=False, all=False,seq_length=126, embedding_dim=300):
        self.num_ins = num_ins
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.random_select = random_select
        self.all = all
    
    def transform(self, input_dict):
        if len(input_dict['text_info'])==0:
            input_dict['label_text'] = np.array([])
            input_dict['gt_text_prompt']  = np.array([])
            input_dict['text_object_id'] = np.array([])
            input_dict['text_token'] = np.array([])
            
            return input_dict
        else:
            text_infos = input_dict['text_info']
            text_infos_id = list(text_infos)
            unique_text_label = []
            for k, v in text_infos.items():
                if v['label_id'] == -100:
                    unique_text_label.append(17)
                else:
                    unique_text_label.append(v['label_id'])
            unique_text_label = np.array(unique_text_label)

            text_token = []
            label_text = []
            text_object_id = []
            select_texts_infos_id = text_infos_id
            sp_pts_mask = torch.as_tensor(input_dict['sp_pts_mask'])
            gt_text_prompt = []
            for i, select_texts_info_id in enumerate(select_texts_infos_id):
                for id in range(len(text_infos[select_texts_info_id]['text'])):
                    token = text_infos[select_texts_info_id]['text_token'][id]
                    text_token.append(token)
                    label_id = text_infos[select_texts_info_id]['label_id']
                    pts_id = text_infos[select_texts_info_id]['pts_id']
                    label_text.append(label_id)
                    text_object_id.append(eval(select_texts_info_id))
                    gt = torch.zeros(sp_pts_mask.shape[0])
                    gt[pts_id] = 1.
                    gt_text_prompt.append(gt)
                    
            gt_text_prompt = torch.stack(gt_text_prompt, dim=0)
            gt_text_prompt = scatter_mean(gt_text_prompt, sp_pts_mask, dim=-1)
            gt_text_prompt = gt_text_prompt > 0.5
                    
            input_dict['label_text'] = np.array(label_text)
            input_dict['gt_text_prompt'] = gt_text_prompt
            input_dict['text_object_id'] = np.array(text_object_id)
            input_dict['text_token'] = np.array(text_token)
            
            return input_dict


@TRANSFORMS.register_module()
class PointPromptTest(BaseTransform):
    def __init__(self, mode='agile', max_num_point=1, samplePoint=False, is_distance=False, size_file=None):
        self.mode=mode # 'agile'
        self.max_num_point = max_num_point
        self.is_distance = is_distance
        self.samplePoint = samplePoint
        if size_file is not None:
            import pickle
            with open(size_file, 'rb') as file:
                self.sizes = pickle.load(file)
    
    def transform(self, input_dict):
        raw_coords = input_dict['points'].coord.contiguous().clone()
        coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
                                                coordinates=raw_coords,
                                                quantization_size=0.05,
                                                return_index=True,
                                                return_inverse=True)
        sp_pts_mask = np.array(input_dict['sp_pts_mask'])
        
        pts_instance_mask = np.array(input_dict['pts_instance_mask']).copy()
        pts_instance_id = np.unique(pts_instance_mask)
        if 'eval_ann_info' in input_dict.keys():
            pts_instance_mask = pts_instance_mask-2
            pts_instance_id = pts_instance_id.copy()
            pts_instance_id = pts_instance_id-2
            pts_instance_id = np.delete(pts_instance_id, np.where(pts_instance_id<0))
        else:
            pts_instance_id = np.delete(pts_instance_id, np.where(pts_instance_id==-1))

        point_prompts = np.array([])
        select_ids = np.array([])
        point_prompt_distance_norms = np.array([])
        select_ids_filter = np.array([])

        if len(pts_instance_id) > 0:
            if self.mode=='agile':
                select_ids = pts_instance_id.copy()
                for select_id in select_ids:
                    gt_inst = (pts_instance_mask == select_id).astype(np.int32)
                    me_coord = raw_coords[unique_map]
                    me_gt_inst = gt_inst[unique_map]
                    valid_index = torch.where(torch.as_tensor(me_gt_inst))[0]
                    zero_indices = (me_gt_inst == 0)  # background
                    one_indices = (me_gt_inst == 1)  # foreground
                    if one_indices.sum() == 0:
                        continue
                    pairwise_distances = torch.cdist(me_coord[zero_indices, :], me_coord[one_indices, :])
                    pairwise_distances, _ = torch.min(pairwise_distances, dim=0)
                    me_index = valid_index[torch.argmax(pairwise_distances)]
                    global_index = unique_map[me_index].numpy().astype(np.int32)
                    assert gt_inst[global_index] == 1
                    point_prompts = np.append(point_prompts, global_index)
                    select_ids_filter = np.append(select_ids_filter, select_id)
        else:
            pass
        if len(select_ids_filter)==0:
            pass
        elif len(select_ids_filter)==1:
            pass
        input_dict['pts_instance_objextId_shuffle'] = input_dict['pts_instance_objextId'][select_ids_filter.astype(int)]    
        input_dict['point_prompts'] = point_prompts
        input_dict['point_prompt_instance_ids'] = select_ids_filter.astype(int)
        input_dict['point_prompt_sp_ids'] = sp_pts_mask[list(point_prompts.astype(int))]
        if self.is_distance:
            input_dict['point_prompt_distance_norms'] = point_prompt_distance_norms

        return input_dict


@TRANSFORMS.register_module()
class AddSuperPointAnnotations(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True,
                 merge_ov=False):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
        self.merge_ov = merge_ov
 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        assert idxs[0] == -1
        input_dict['pts_instance_objextId'] = idxs[1:] - 1
        
        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1
        pts_instance_mask = mapping[pts_instance_mask]
        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()

        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()
        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        insts = new_idxs[1:] - 1
        if self.merge_ov:
            novel_sp_masks = input_dict['sam3d_pseudo_sp_masks']
            is_novel = torch.zeros(len(insts) + len(novel_sp_masks) + self.num_classes + 1)
            is_novel[len(insts):len(insts)+len(novel_sp_masks)] = 1
            input_dict['is_novel'] = is_novel.bool()
            sp_masks_inst = torch.cat((sp_masks_inst, novel_sp_masks), dim=0)
        
        num_stuff_cls = len(self.stuff_classes)
        
        if self.merge_non_stuff_cls:
            if self.merge_ov:
                gt_labels = insts.new_zeros(len(insts) + len(novel_sp_masks) + num_stuff_cls + 1)
            else:
                gt_labels = insts.new_zeros(len(insts) + num_stuff_cls + 1)
        else:
            if self.merge_ov:
                gt_labels = insts.new_zeros(len(insts) + len(novel_sp_masks) + self.num_classes + 1)
            else:
                gt_labels = insts.new_zeros(len(insts) + self.num_classes + 1)

        for inst in insts:
            index = pts_semantic_mask[pts_instance_mask == inst][0]
            gt_labels[inst] = index - num_stuff_cls
        
        input_dict['gt_labels_3d'] = gt_labels.numpy()

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        sem_mask = torch.nn.functional.one_hot(sem_mask, 
                                    num_classes=self.num_classes + 1)
       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
        
        if self.merge_non_stuff_cls:
            sp_masks_seg = torch.vstack((
                sp_masks_seg[:num_stuff_cls, :], 
                sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))
        
        sp_masks_all = torch.vstack((sp_masks_inst, sp_masks_seg))

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict
