from cog import BasePredictor, Input, Path
import torch
from detectron2.data import MetadataCatalog
from PIL import Image, ImageFilter
import detectron2.data.transforms as T
import numpy as np
import pickle
import imutils
import cv2
# Import libraries
import numpy as np
import torch
from demo.visualizer import _PanopticPrediction
# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from demo.defaults import DefaultPredictor
import numpy.ma as ma
from demo.visualizer import Visualizer, ColorMode
from typing import List
# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
import pillow_avif
from pillow_heif import register_heif_opener

register_heif_opener()
cpu_device = torch.device("cpu")
SWIN_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_swin_large_bs16_160k_896x896.yaml",}

DINAT_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}

def setup_cfg(dataset, model_path, use_swin):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    # add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    if use_swin:
      cfg_path = SWIN_CFG_DICT[dataset]
    else:
      cfg_path = DINAT_CFG_DICT[dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg

def setup_modules(dataset, model_path, use_swin):
    cfg = setup_cfg(dataset, model_path, use_swin)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )

    return predictor, metadata

exceptions = ["sky", "floor", "ceiling", "road, route", "grass", "earth, ground", "field", "sand", "hill", "fireplace", "land, ground, soil", "wall", "ceiling", "door", "mountain, mount", "curtain", "water", "sea", "path", "countertop", "bench", "shelf", "dirt track", "stage", "lake", "screen", ]

def merge_og_image_and_mask(img, background):
    bg_blur = background.convert("L")
    bg_blur2 = bg_blur.filter(ImageFilter.BoxBlur(1))
    wip_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
    return Image.composite(img, wip_img, mask=bg_blur2)

def parse_panoptic_seg_masks(panoptic_seg_output, metadata, stuff_mapper):
    panoptic_seg, segments_info = panoptic_seg_output
    pred = _PanopticPrediction(panoptic_seg.cpu(), segments_info, metadata)
    semantic_masks = list(pred.semantic_masks())
    instance_masks = list(pred.instance_masks())
    panoptic_preds = semantic_masks + instance_masks
    print([preds[1] for preds in panoptic_preds])
    mask_classes = [preds[1] for preds in panoptic_preds]
    if len(mask_classes) == 0 or len(mask_classes) == 1:
        return None, None
    valid_segments = []
    for segment, sinfo in panoptic_preds:
        if stuff_mapper[sinfo["category_id"]] in exceptions:
            continue
        valid_segments.append(segment.astype("uint8"))
    mask_merged = None
    if len(valid_segments) == 1:
        mask_merged = valid_segments[0].astype("bool")
    else:
        for idx, segment in enumerate(valid_segments):
            if idx == 0:
                mask_merged = segment
            if len(valid_segments) == idx + 1:
                break
            mask_merged = ma.mask_or(mask_merged.astype("uint8"), valid_segments[idx + 1])
    background = Image.fromarray(mask_merged)
    return background, mask_merged

def parse_semantic_seg_masks(semantic_seg_output, metadata):
    sem_seg = semantic_seg_output.argmax(dim=0).to(cpu_device)
    if isinstance(sem_seg, torch.Tensor):
        sem_seg = sem_seg.numpy()
    labels, areas = np.unique(sem_seg, return_counts=True)
    sorted_idxs = np.argsort(-areas).tolist()
    labels = labels[sorted_idxs]

    validated_masks = []
    for label in filter(lambda l: l < len(metadata.stuff_classes), labels):
        binary_mask = (sem_seg == label).astype(np.uint8)
        text = metadata.stuff_classes[label]
        validated_masks.append({"mask": binary_mask, "class": text})

    valid_segments = []
    for mask_info in validated_masks:
        if mask_info["class"] in exceptions:
            continue
        valid_segments.append(mask_info["mask"].astype("uint8"))
    mask_merged = None
    if len(valid_segments) == 1:
        mask_merged = valid_segments[0].astype("bool")
    else:
        for idx, segment in enumerate(valid_segments):
            if idx == 0:
                mask_merged = segment
            if len(valid_segments) == idx + 1:
                break
            mask_merged = ma.mask_or(mask_merged.astype("uint8"), valid_segments[idx + 1])
    background = Image.fromarray(mask_merged)
    return background, mask_merged

def save_final_img(img, background, mask_merged, task_type):
    if mask_merged.size:
        img1 = merge_og_image_and_mask(img=img, background=background)
        img1.save(f"{task_type}_output.png")
    else:
        img.save(f"{task_type}_output.png")
class Predictor(BasePredictor):
    def setup(self):
        self.predictor, self.metadata = setup_modules("ade20k", "896x896_250_16_swin_l_oneformer_ade20k_160k.pth", True)
        self.stuff_mapper = dict(zip(range(0, len(self.metadata.stuff_classes)), self.metadata.stuff_classes))
        print("Model loaded...")

    def predict(
        self,
        image: Path = Input(description="Upload image in following formats: jpg, jpeg, png, heic, webp, heif, avif"),
        task_type: str = Input(description="Add panoptic or semantic or all"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        img = Image.open(image).convert("RGB")
        if img.format in ["AVIF", "HEIC"]:
            img = img.convert("RGB")
        output = self.predictor(np.asarray(img), task_type)
        background = None
        mask_merged = None
        if task_type == "panoptic":
            background, mask_merged = parse_panoptic_seg_masks(panoptic_seg_output=output["panoptic_seg"], metadata=self.metadata, stuff_mapper=self.stuff_mapper)
            if mask_merged is None and background is None:
                img.save(f"{task_type}_output.png")
                return [Path(f"{task_type}_output.png")]
            save_final_img(img=img, background=background, mask_merged=mask_merged, task_type=task_type)
            return [Path(f"{task_type}_output.png")]
        elif task_type == "semantic":
            background, mask_merged = parse_semantic_seg_masks(semantic_seg_output=output["sem_seg"], metadata=self.metadata)
            save_final_img(img=img, background=background, mask_merged=mask_merged, task_type=task_type)
            return [Path(f"{task_type}_output.png")]
        elif task_type == "all":
            background = mask_merged = None
            background, mask_merged = parse_semantic_seg_masks(semantic_seg_output=output["sem_seg"], metadata=self.metadata)
            save_final_img(img=img, background=background, mask_merged=mask_merged, task_type="semantic")

            background = mask_merged = None
            background, mask_merged = parse_panoptic_seg_masks(panoptic_seg_output=output["panoptic_seg"], metadata=self.metadata, stuff_mapper=self.stuff_mapper)
            if mask_merged is None and background is None:
                img.save(f"panoptic_output.png")
            else:
                save_final_img(img=img, background=background, mask_merged=mask_merged, task_type="panoptic")
            return [Path(f"semantic_output.png"), Path(f"panoptic_output.png")]

