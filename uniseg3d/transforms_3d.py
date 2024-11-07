import numpy as np
import scipy
import torch
import random
from torch_scatter import scatter_mean
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

import MinkowskiEngine as ME


@TRANSFORMS.register_module()
class TextPromptGeneration(BaseTransform):
    def __init__(self, num_ins = 3, random_select=False, seq_length=126, embedding_dim=300):
        self.num_ins = num_ins
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.random_select = random_select
    
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
            if self.random_select:
                num_ins = np.random.randint(len(text_infos_id), size=1)[0] + 1
            else:
                num_ins = len(text_infos_id)
                
            text_token = []
            label_text = np.zeros((num_ins,))
            text_object_id = np.zeros((num_ins,))
            select_texts_infos_id = random.sample(text_infos_id, num_ins)
            gt_sp_masks = torch.as_tensor(input_dict['gt_sp_masks'])
            gt_text_prompt = torch.zeros((num_ins, gt_sp_masks.shape[1]))
            pts_instance_objextId = input_dict['pts_instance_objextId']
            for i, select_texts_info_id in enumerate(select_texts_infos_id):
                id = random.sample(range(len(text_infos[select_texts_info_id]['text'])), 1)[0]
                token = text_infos[select_texts_info_id]['text_token'][id]
                text_token.append(token)
                label_id = text_infos[select_texts_info_id]['label_id']
                label_text[i] = label_id
                text_object_id[i] = eval(select_texts_info_id)
                object_id = torch.tensor(eval(select_texts_info_id))
                id = torch.where(pts_instance_objextId == object_id)[0]
                assert len(id)>0
                gt_text_prompt[i] = gt_sp_masks[id[0]]
                
            input_dict['label_text'] = label_text
            input_dict['gt_text_prompt']  = gt_text_prompt
            input_dict['text_object_id'] = text_object_id
            input_dict['text_token'] = np.array(text_token)
            
            return input_dict


@TRANSFORMS.register_module()
class TextPromptTest(BaseTransform):
    def __init__(self, num_ins = 3, seq_length=126, embedding_dim=300):
        self.num_ins = num_ins
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

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
class PointPromptGeneration(BaseTransform):
    def __init__(self, samplePoint=False):
        self.samplePoint = samplePoint

    def transform(self, input_dict):
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

        if len(pts_instance_id) > 0:
            select_ids = pts_instance_id.copy()
            np.random.shuffle(select_ids)   
            for select_id in select_ids:
                pts_select_id = np.where(pts_instance_mask == select_id)[0]
                if self.samplePoint:
                    instance_points = input_dict['points'][pts_select_id][:,:3]
                    if len(instance_points)>500:
                        sample_ratio = 500.0/len(instance_points)
                    else:
                        sample_ratio = 1.0

                    sample_size = int(len(instance_points) * sample_ratio)

                    sample_indices = np.random.choice(len(instance_points), sample_size, replace=False)
                    sampled_points = instance_points[sample_indices]
                    centroid = np.mean(sampled_points, axis=0)
                    distances = np.linalg.norm(sampled_points - centroid, axis=1)
                    distance_order = np.argsort(distances) 
                    ordered_pts_select_id = pts_select_id[sample_indices][distance_order[0]]
                    point_prompts = np.append(point_prompts, ordered_pts_select_id)
                else:
                    random_pts_select_id = np.random.choice(pts_select_id, size=1, replace=False)
                    point_prompts = np.append(point_prompts, random_pts_select_id)
        else:
            pass

        input_dict['pts_instance_objextId_shuffle'] = input_dict['pts_instance_objextId'][select_ids]    
        input_dict['point_prompts'] = point_prompts
        input_dict['point_prompt_instance_ids'] = select_ids
        input_dict['point_prompt_sp_ids'] = sp_pts_mask[list(point_prompts.astype(int))]

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
class ElasticTransfrom(BaseTransform):
    """Apply elastic augmentation to a 3D scene. Required Keys:

    Args:
        gran (List[float]): Size of the noise grid (in same scale[m/cm]
            as the voxel grid).
        mag (List[float]): Noise multiplier.
        voxel_size (float): Voxel size.
        p (float): probability of applying this transform.
    """

    def __init__(self, gran, mag, voxel_size, p=1.0):
        self.gran = gran
        self.mag = mag
        self.voxel_size = voxel_size
        self.p = p

    def transform(self, input_dict):
        """Private function-wrapper for elastic transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after elastic, 'points' is updated
            in the result dict.
        """
        coords = input_dict['points'].tensor[:, :3].numpy() / self.voxel_size
        if np.random.rand() < self.p:
            coords = self.elastic(coords, self.gran[0], self.mag[0])
            coords = self.elastic(coords, self.gran[1], self.mag[1])
        input_dict['elastic_coords'] = coords
        return input_dict

    def elastic(self, x, gran, mag):
        """Private function for elastic transform to a points.

        Args:
            x (ndarray): Point cloud.
            gran (List[float]): Size of the noise grid (in same scale[m/cm]
                as the voxel grid).
            mag: (List[float]): Noise multiplier.
        
        Returns:
            dict: Results after elastic, 'points' is updated
                in the result dict.
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        noise_dim = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(noise_dim[0], noise_dim[1],
                            noise_dim[2]).astype('float32') for _ in range(3)
        ]

        for blur in [blur0, blur1, blur2, blur0, blur1, blur2]:
            noise = [
                scipy.ndimage.filters.convolve(
                    n, blur, mode='constant', cval=0) for n in noise
            ]

        ax = [
            np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in noise_dim
        ]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0) for n in noise
        ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * mag


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
