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
            "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",}

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

class Predictor(BasePredictor):
    def setup(self):
        self.predictor, self.metadata = setup_modules("ade20k", "250_16_swin_l_oneformer_ade20k_160k.pth", True)
        self.stuff_mapper = dict(zip(range(0, len(self.metadata.stuff_classes)), self.metadata.stuff_classes))
        print("Model loaded...")

    def predict(
        self,
        image: Path = Input(description="Upload image in following formats: jpg, jpeg, png, heic, webp, heif, avif"),
        show_all_masks: str = Input(description="Add 'yes' for showing all the masks")
    ) -> List[Path]:
        """Run a single prediction on the model"""
        img = Image.open(image).convert("RGB")
        if img.format in ["AVIF", "HEIC"]:
            img = img.convert("RGB")
        output = self.predictor(np.asarray(img), "panoptic")
        panoptic_seg, segments_info = output["panoptic_seg"]
        pred = _PanopticPrediction(panoptic_seg.cpu(), segments_info, self.metadata)
        semantic_masks = list(pred.semantic_masks())
        instance_masks = list(pred.instance_masks())
        panoptic_preds = semantic_masks + instance_masks
        print([preds[1] for preds in panoptic_preds])
        valid_segments = []
        for segment, sinfo in panoptic_preds:
            if self.stuff_mapper[sinfo["category_id"]] in exceptions:
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

        if mask_merged.size:
            bg_blur = background.convert("L")
            bg_blur2 = bg_blur.filter(ImageFilter.BoxBlur(1))
            wip_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
            img1 = Image.composite(img, wip_img, mask=bg_blur2)
        
            img1.save("output.png")
        else:
            img.save("output.png")
        # CV2 image read
        if show_all_masks == "yes":
            cv2_img = cv2.imread(str(image))
            visualizer = Visualizer(cv2_img[:, :, ::-1], metadata=self.metadata, instance_mode=ColorMode.IMAGE)
            out = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(cpu_device), segments_info, alpha=0.5
            )
            all_seg_img = Image.fromarray(out.get_image())
            all_seg_img.save("all_seg.png")
            return [Path("output.png"), Path("all_seg.png")]

        return [Path("output.png")]
