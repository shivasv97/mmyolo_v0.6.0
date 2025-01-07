# _base_ = './yolov8_l_syncbn_fast_8xb16-500e_vindr.py'
_base_ = [ './yolov8_x_syncbn_fast_8xb16.py', '../_base_/med_datasets/vindr.py']

# deepen_factor = 1.00
# widen_factor = 1.25

# model = dict(
#     backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
#     neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
#     bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
n=14
model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=n
        )
    ),
    train_cfg = dict(
        assigner = dict(
            num_classes=n
        )
    )
)

load_from = "/scratch/ssenth21/bench_boost_lclz/mmyolo_v0.6.0/checkpoints/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth"
# load_from = "/anvil/projects/x-med220025/ssenth21/bench_boost_lclz/mmyolo_v0.6.0/checkpoints/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth"


# # ========================Frequently modified parameters======================
# # -----data related-----
# dataset_type = 'YOLOv5CocoDataset'
# data_root = '/scratch/ssenth21/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/'  # Root path of data
# # data_root = '/anvil/projects/x-med220025/datasets/vindr_cxr/1.0.0/'  # Root path of data
# # Path of train annotation file
# train_ann_file = 'annotations/VinDrCXR_Kaggle_14Diseases_TRAIN.json'
# train_data_prefix = 'train_jpeg/'  # Prefix of train image path
# # Path of val annotation file
# val_ann_file = 'annotations/VinDrCXR_Kaggle_14Diseases_TEST.json'
# val_data_prefix = 'test_jpeg/'  # Prefix of val image path
# val_batch_size_per_gpu = 1
# # Worker to pre-fetch data for each single GPU during validation
# val_num_workers = 2

# # Config of batch shapes. Only on val.
# # We tested YOLOv8-m will get 0.02 higher than not using it.
# batch_shapes_cfg = None
# img_scale = (1024, 1024)  # width, height

# class_name = ( 'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis')
# num_classes = len(class_name)  # Number of classes for classification
# metainfo = dict(classes=class_name)

# train_batch_size_per_gpu = 2
# # train_batch_size_per_gpu = 16
# # Worker to pre-fetch data for each single GPU during training
# train_num_workers = 8
# # persistent_workers must be False if num_workers is 0
# persistent_workers = True

# # --------------------------------
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
#     dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
#     dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(
#         type='mmdet.RandomChoice',
#         transforms=[[
#             dict(
#                 type='mmdet.RandomChoiceResize',
#                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                         (736, 1333), (768, 1333), (800, 1333)],
#                 keep_ratio=True)
#         ],
#                     [
#                         dict(
#                             type='mmdet.RandomChoiceResize',
#                             scales=[(400, 1333), (500, 1333), (600, 1333)],
#                             keep_ratio=True),
#                         dict(
#                             type='mmdet.RandomCrop',
#                             crop_type='absolute_range',
#                             crop_size=(384, 600),
#                             allow_negative_crop=True),
#                         dict(
#                             type='mmdet.RandomChoiceResize',
#                             scales=[(480, 1333), (512, 1333), (544, 1333),
#                                     (576, 1333), (608, 1333), (640, 1333),
#                                     (672, 1333), (704, 1333), (736, 1333),
#                                     (768, 1333), (800, 1333)],
#                             keep_ratio=True)
#                     ]]),
#     dict(type='mmdet.PackDetInputs')
# ]

# # test_pipeline = [
# #     dict(type='LoadImageFromFile', backend_args=backend_args),
# #     dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
# #     dict(type='LoadAnnotations', with_bbox=True),
# #     dict(
# #         type='PackDetInputs',
# #         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
# #                    'scale_factor'))
# # ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
#     dict(type='mmdet.Resize', scale=(1333, 800), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

# # # ----------------------------------------
# # pre_transform = [
# #     dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
# #     dict(type='LoadAnnotations', with_bbox=True)
# # ]

# # albu_train_transforms = [
# #     dict(type='Blur', p=0.01),
# #     dict(type='MedianBlur', p=0.01),
# #     dict(type='ToGray', p=0.01),
# #     dict(type='CLAHE', p=0.01)
# # ]

# # last_transform = [
# #     dict(
# #         type='mmdet.Albu',
# #         transforms=albu_train_transforms,
# #         bbox_params=dict(
# #             type='BboxParams',
# #             format='pascal_voc',
# #             label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
# #         keymap={
# #             'img': 'image',
# #             'gt_bboxes': 'bboxes'
# #         }),
# #     dict(type='YOLOv5HSVRandomAug'),
# #     dict(type='mmdet.RandomFlip', prob=0.5),
# #     dict(
# #         type='mmdet.PackDetInputs',
# #         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
# #                    'flip_direction'))
# # ]

# # train_pipeline = [
# #     *pre_transform,
# #     dict(
# #         type='Mosaic',
# #         img_scale=img_scale,
# #         pad_val=114.0,
# #         pre_transform=pre_transform),
# #     dict(
# #         type='YOLOv5RandomAffine',
# #         max_rotate_degree=0.0,
# #         max_shear_degree=0.0,
# #         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
# #         max_aspect_ratio=max_aspect_ratio,
# #         # img_scale is (width, height)
# #         border=(-img_scale[0] // 2, -img_scale[1] // 2),
# #         border_val=(114, 114, 114)),
# #     *last_transform
# # ]

# train_dataloader = dict(
#     _delete_=True,
#     batch_size=train_batch_size_per_gpu,
#     num_workers=train_num_workers,
#     persistent_workers=persistent_workers,
#     pin_memory=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     collate_fn=dict(type='yolov5_collate'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file=train_ann_file,
#         data_prefix=dict(img=train_data_prefix),
#         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#         pipeline=train_pipeline))

# # test_pipeline = [
# #     dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
# #     dict(type='YOLOv5KeepRatioResize', scale=img_scale),
# #     dict(
# #         type='LetterResize',
# #         scale=img_scale,
# #         allow_scale_up=False,
# #         pad_val=dict(img=114)),
# #     dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
# #     dict(
# #         type='mmdet.PackDetInputs',
# #         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
# #                    'scale_factor', 'pad_param'))
# # ]

# val_dataloader = dict(
#     _delete_=True,
#     batch_size=val_batch_size_per_gpu,
#     num_workers=val_num_workers,
#     persistent_workers=persistent_workers,
#     pin_memory=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         metainfo=metainfo,
#         data_root=data_root,
#         test_mode=True,
#         data_prefix=dict(img=val_data_prefix),
#         ann_file=val_ann_file,
#         pipeline=test_pipeline,
#         batch_shapes_cfg=batch_shapes_cfg))

# test_dataloader = val_dataloader