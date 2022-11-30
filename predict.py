from cog import BasePredictor, Input, Path
import torch
from detectron2.data import MetadataCatalog
from PIL import Image
import detectron2.data.transforms as T
import numpy as np
import pickle
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="oneformer")

# Import libraries
import numpy as np
import torch

# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from demo.defaults import DefaultPredictor


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
        self.predictor, _ = setup_modules("ade20k", "250_16_dinat_l_oneformer_ade20k_160k.pth", True)
        """Load the model into memory to make running multiple predictions efficient"""
        print("Model loaded...")

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        im = Image.open(image)
        # processed_input = preprocess(im)
        output = self.predictor(im, "panoptic")
        print(output["panoptic_seg"])
        with open("output_dump.pkl", "wb") as pkl_dump:
            pickle.dump(output["panoptic_seg"], pkl_dump)
        return "output_dump.pkl"
