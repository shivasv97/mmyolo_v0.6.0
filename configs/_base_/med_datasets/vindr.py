_base_=['../det_p5_tta.py', ]
dataset_type = 'YOLOv5CocoDataset'
# data_root = '/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/'  # Root path of data
data_root = '/scratch/ssenth21/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/'  # Root path of data
# data_root = '/anvil/projects/x-med220025/datasets/vindr_cxr/1.0.0/'  # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/VinDrCXR_Kaggle_14Diseases_TRAIN.json'
train_data_prefix = 'train_jpeg/'  # Prefix of train image path

val_ann_file = 'annotations/VinDrCXR_Kaggle_14Diseases_TEST.json'
val_data_prefix = 'test_jpeg/'  # Prefix of val image path
test_ann_file = 'annotations/VinDrCXR_Kaggle_14Diseases_TEST.json'
test_data_prefix = 'test_jpeg/'  # Prefix of val image path
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# # Path of val annotation file
# val_ann_file = 'annotations/VinDrCXR_Kaggle_14Diseases_TEST.json'
# val_data_prefix = 'test_jpeg/'  # Prefix of val image path
# val_batch_size_per_gpu = 1
# # Worker to pre-fetch data for each single GPU during validation
# val_num_workers = 2

# Config of batch shapes. Only on val.
# We tested YOLOv8-m will get 0.02 higher than not using it.
batch_shapes_cfg = None
# img_scale = (1333, 800)  # width, height
img_scale = (1024, 1024)  # width, height

class_name = ( 'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis')
num_classes = len(class_name)  # Number of classes for classification
metainfo = dict(classes=class_name)

train_batch_size_per_gpu = 2
# train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# --------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'), # , backend_args=_base_.backend_args
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    
    dict(
        type='mmdet.RandomChoice',
        transforms=[[
            dict(
                type='mmdet.RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='mmdet.RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='mmdet.RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='mmdet.RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(
        type='mmdet.FixShapeResize',
        width=img_scale[0],
        height=img_scale[1],
        ),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'), # , backend_args=_base_.backend_args
    dict(type='mmdet.Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='mmdet.FixShapeResize',
        width=img_scale[0],
        height=img_scale[1],
        ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=test_data_prefix),
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + test_ann_file,
    metric='bbox')