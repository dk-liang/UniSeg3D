# adapted from mmdet3d/models/data_preprocessors/data_preprocessor.py
from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
from mmdet3d.registry import MODELS


@MODELS.register_module()
class Det3DDataPreprocessor_(Det3DDataPreprocessor):
    def simple_process(self, data, training=False):
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict

        if 'elastic_coords' in inputs:
            batch_inputs['elastic_coords'] = inputs['elastic_coords']

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if hasattr(self, 'boxtype2tensor') and self.boxtype2tensor:
                    from mmdet.models.utils.misc import \
                        samplelist_boxtype2tensor
                    samplelist_boxtype2tensor(data_samples)
                elif hasattr(self, 'boxlist2tensor') and self.boxlist2tensor:
                    from mmdet.models.utils.misc import \
                        samplelist_boxlist2tensor
                    samplelist_boxlist2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)

                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        if 'point_prompts' in inputs:
            batch_inputs['point_prompts'] = inputs['point_prompts']

        if 'point_prompt_instance_ids' in inputs:
            batch_inputs['point_prompt_instance_ids'] = inputs['point_prompt_instance_ids']

        if 'point_prompt_sp_ids' in inputs:
            batch_inputs['point_prompt_sp_ids'] = inputs['point_prompt_sp_ids']
        
        if 'emb_text' in inputs:
            batch_inputs['emb_text'] = inputs['emb_text']

        if 'label_text' in inputs:
            batch_inputs['label_text'] = inputs['label_text']

        if 'gt_text_prompt' in inputs:
            batch_inputs['gt_text_prompt'] = inputs['gt_text_prompt']
            
        if 'text_token' in inputs:
            batch_inputs['text_token'] = inputs['text_token']
            
        if 'text_object_id' in inputs:
            batch_inputs['text_object_id'] = inputs['text_object_id']
        
        if 'pts_instance_objextId_shuffle' in inputs:
            batch_inputs['pts_instance_objextId_shuffle'] = inputs['pts_instance_objextId_shuffle']

        if 'point_prompt_distance_norms' in inputs:
            batch_inputs['point_prompt_distance_norms'] = inputs['point_prompt_distance_norms']

        return {'inputs': batch_inputs, 'data_samples': data_samples}
