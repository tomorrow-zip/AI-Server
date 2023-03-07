import bentoml
import torch.cuda

from mmdet.apis import init_detector


config_file = (
    "checkpoints/detector.config.py"
)
weight_file = (
    "checkpoints/"
    "mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth"
)

model = init_detector(
    config_file,
    weight_file,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

save_model = bentoml.pytorch.save_model(
    name="detection_and_segmentation",
    model=model,
    custom_objects={"classes": model.CLASSES},
)
