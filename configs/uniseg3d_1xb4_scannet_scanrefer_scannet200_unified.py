_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/datasets/scannet-seg.py'
]
custom_imports = dict(imports=['uniseg3d'])
find_unused_parameters = True

# model settings
num_channels = 32
num_instance_classes = 18
num_semantic_classes = 20

pred_iou = True
use_pseudo_cls_supervise = True
inst_weight = 1.0
hype_lambda = 1.0

model = dict(
    type='UniSeg3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.02,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    query_thr=0.5,
    inst_test_iou = False,
    pano_test_iou = True,
    pred_iou = pred_iou,
    set_query_mask=True,
    set_all_mask=True,
    is_type_embedding=True,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    lang = dict(
        type='LangModule',
        input_channels=768,
        model_name = 'convnext_large_d_320', 
        init_cfg=dict(type='clip_pretrain', 
                checkpoint='work_dirs/pretrained/convnext_large_d_320/open_clip_pytorch_model.bin'),
        fix=True,
        out_features = 32,),
    decoder=dict(
        type='UnifiedQueryDecoder',
        num_layers=6,
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False,
        sphere_cls = True,
        vocabulary_cls_embedding_path = 'data/scannet/scannet_cls_embedding.pth'),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=600,
        inst_score_thr=0.0,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1]))

# dataset settings
dataset_type = 'UnifiedSegDataset'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(type='PointSegClassMapping'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5]),
            dict(
                type='AddSuperPointAnnotations',
                num_classes=num_semantic_classes,
                stuff_classes=[0, 1],
                merge_non_stuff_cls=False),
            dict(type='TextPromptTest',
                num_ins=3,
                random_select=False,
                all = True,
                seq_length=126, 
                embedding_dim=300,),
            dict(
                type='PointPromptTest',
                max_num_point=3),
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask', 'gt_sp_masks', 'gt_labels_3d',
                                        'point_prompts', 'point_prompt_instance_ids', 'point_prompt_sp_ids',
                                        'label_text', 'gt_text_prompt','text_token','text_object_id',
                                        'pts_instance_objextId_shuffle'
                                        ])
]

# run settings
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='uniseg3d_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        ignore_index=num_semantic_classes,
        test_mode=True))
test_dataloader = val_dataloader

class_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
    'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
    'bathtub', 'otherfurniture']
class_names += ['unlabeled']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names,
    dataset_name='ScanNet')

sem_mapping = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
inst_mapping = sem_mapping[2:]
val_evaluator = dict(
    type='PromptSupportedUnifiedSegMetric',
    stuff_class_inds=[0, 1], 
    thing_class_inds=list(range(2, num_semantic_classes)), 
    min_num_points=1, 
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

epoch = 512

param_scheduler = dict(type='PolyLR', begin=0, end=epoch, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3),
    )

default_hooks = dict(
    checkpoint=dict(interval=8,
                    max_keep_ckpts=10,
                    save_best=['all_ap', 'miou', 'pq'],
                    rule='greater'),
    logger=dict(type='LoggerHook', interval=50),)

randomness = dict(
    seed=1,
    diff_rank_seed=True,
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=epoch,
    dynamic_intervals=[(1, 16), (epoch - 16, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')