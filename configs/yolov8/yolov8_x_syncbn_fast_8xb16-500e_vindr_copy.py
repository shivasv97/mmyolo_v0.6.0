_base_ = './yolov8_l_syncbn_fast_8xb16-500e_vindr.py'

deepen_factor = 1.00
widen_factor = 1.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
load_from = "/scratch/ssenth21/bench_boost_lclz/mmyolo_v0.6.0/checkpoints/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth"