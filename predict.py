from cog import BasePredictor, Input, Path
import torch
from detectron2.data import MetadataCatalog
from PIL import Image, ImageFilter
import detectron2.data.transforms as T
import numpy as np
import pickle
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="oneformer")
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


# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

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

# def preprocess_cv2(image):
exceptions = ["sky", "floor", "ceiling", "road, route", "grass", "earth, ground", "field", "sand", "hill"]
def preprocess(image):
    image = np.asarray(image)
    height, width = image.shape[:2]

    print("  > Preprocessing image...")
    aug = T.ResizeShortestEdge(
        [640, 640], 2560
    )
    image = aug.get_transform(image).apply_image(image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width, "task": "panoptic"}]
    return inputs

class Predictor(BasePredictor):
    def setup(self):
        self.predictor, self.metadata = setup_modules("ade20k", "250_16_swin_l_oneformer_ade20k_160k.pth", True)
        self.stuff_mapper = dict(zip(range(0, len(self.metadata.stuff_classes)), self.metadata.stuff_classes))
        print("Model loaded...")

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        img = Image.open(image)

        output = self.predictor(np.asarray(img), "panoptic")
        panoptic_seg, segments_info = output["panoptic_seg"]
        pred = _PanopticPrediction(panoptic_seg.cpu(), segments_info, self.metadata)
        panoptic_preds = list(pred.semantic_masks())
        valid_segments = []
        for segment, sinfo in panoptic_preds:
            if self.stuff_mapper[sinfo["category_id"]] in exceptions:
                continue
            valid_segments.append(segment.astype("uint8"))
        mask_merged = None
        for idx, segment in enumerate(valid_segments):
            if idx == 0:
                mask_merged = segment
            if len(valid_segments) == idx + 1:
                break
            mask_merged = ma.mask_or(mask_merged.astype("uint8"), valid_segments[idx + 1])
        background = Image.fromarray(mask_merged)
        bg_blur = background.convert("L")
        bg_blur2 = bg_blur.filter(ImageFilter.BoxBlur(1))
        wip_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        img1 = Image.composite(img, wip_img, mask=bg_blur2)
        img1.save("output.png")
        return Path("output.png")
