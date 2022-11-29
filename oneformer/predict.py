# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from datetime import datetime
from cog import BasePredictor, Input, Path
import detectron2
from detectron2.utils.logger import setup_logger
import gdown
import base64
from io import BytesIO
# Import libraries
import numpy as np
import cv2
import torch
import numpy.ma as ma
# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from PIL import Image, ImageFilter
import detectron2.data.transforms as T

# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

dataset = "ade20k"
cpu_device = torch.device("cpu")
SWIN_CFG_DICT = {"cityscapes": "./configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
            "coco": "./configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
            "ade20k": str(CONFIG_PATH / "oneformer_swin_large_IN21k_384_bs16_160k.yaml"),}

DINAT_CFG_DICT = {"cityscapes": "./configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
            "coco": "./configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
            "ade20k": "./configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}

exceptions = ["sky", "floor", "ceiling", "road, route", "grass", "earth, ground", "field", "sand", "hill"]

model_path = Path("image-bg-removal.pkl") 
if not model_path.exists():
    print("Downloading model from gdrive...")
    gdrive_url = "https://drive.google.com/uc?id=1AZZNSnfJXI-GJeeicMv8vccSeLTPDtEK"
    gdown.download(gdrive_url, str(model_path))

def add_common_config(cfg):
    """
    Add config for common configuration
    """

    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "oneformer_unified"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.INPUT.TASK_SEQ_LEN = 77
    cfg.INPUT.MAX_SEQ_LEN = 77

    cfg.INPUT.TASK_PROB = CN()
    cfg.INPUT.TASK_PROB.SEMANTIC = 0.33
    cfg.INPUT.TASK_PROB.INSTANCE = 0.66

    # test dataset
    cfg.DATASETS.TEST_PANOPTIC = ("",)
    cfg.DATASETS.TEST_INSTANCE = ("",)
    cfg.DATASETS.TEST_SEMANTIC = ("",)

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "unified_dense_recognition"
    cfg.WANDB.NAME = None

    cfg.MODEL.IS_TRAIN = False
    cfg.MODEL.IS_DEMO = True

    # text encoder config
    cfg.MODEL.TEXT_ENCODER = CN()

    cfg.MODEL.TEXT_ENCODER.WIDTH = 256
    cfg.MODEL.TEXT_ENCODER.CONTEXT_LENGTH = 77
    cfg.MODEL.TEXT_ENCODER.NUM_LAYERS = 12
    cfg.MODEL.TEXT_ENCODER.VOCAB_SIZE = 49408
    cfg.MODEL.TEXT_ENCODER.PROJ_NUM_LAYERS = 2
    cfg.MODEL.TEXT_ENCODER.N_CTX = 16

    # mask_former inference config
    cfg.MODEL.TEST = CN()
    cfg.MODEL.TEST.SEMANTIC_ON = True
    cfg.MODEL.TEST.INSTANCE_ON = False
    cfg.MODEL.TEST.PANOPTIC_ON = False
    cfg.MODEL.TEST.DETECTION_ON = False
    cfg.MODEL.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.TEST.TASK = "panoptic"

    # TEST AUG Slide
    cfg.TEST.AUG.IS_SLIDE = False
    cfg.TEST.AUG.CROP_SIZE = (640, 640)
    cfg.TEST.AUG.STRIDE = (426, 426)
    cfg.TEST.AUG.SCALE = (2048, 640)
    cfg.TEST.AUG.SETR_MULTI_SCALE = True
    cfg.TEST.AUG.KEEP_RATIO = True
    cfg.TEST.AUG.SIZE_DIVISOR = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.SEM_EMBED_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.INST_EMBED_DIM = 256

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

def add_oneformer_config(cfg):
    """
    Add config for ONE_FORMER.
    """

    # mask_former model config
    cfg.MODEL.ONE_FORMER = CN()

    # loss
    cfg.MODEL.ONE_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.ONE_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.ONE_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.ONE_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.ONE_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.ONE_FORMER.CONTRASTIVE_WEIGHT = 0.5
    cfg.MODEL.ONE_FORMER.CONTRASTIVE_TEMPERATURE = 0.07

    # transformer config
    cfg.MODEL.ONE_FORMER.NHEADS = 8
    cfg.MODEL.ONE_FORMER.DROPOUT = 0.1
    cfg.MODEL.ONE_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.ONE_FORMER.ENC_LAYERS = 0
    cfg.MODEL.ONE_FORMER.CLASS_DEC_LAYERS = 2
    cfg.MODEL.ONE_FORMER.DEC_LAYERS = 6
    cfg.MODEL.ONE_FORMER.PRE_NORM = False

    cfg.MODEL.ONE_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES = 120
    cfg.MODEL.ONE_FORMER.NUM_OBJECT_CTX = 16
    cfg.MODEL.ONE_FORMER.USE_TASK_NORM = True

    cfg.MODEL.ONE_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.ONE_FORMER.ENFORCE_INPUT_PROJ = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.ONE_FORMER.SIZE_DIVISIBILITY = 32

    # transformer module
    cfg.MODEL.ONE_FORMER.TRANSFORMER_DECODER_NAME = "ContrastiveMultiScaleMaskedTransformerDecoder"

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.ONE_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.ONE_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.ONE_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

def add_swin_config(cfg):
    """
    Add config forSWIN Backbone.
    """

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    ## Semask additions
    cfg.MODEL.SWIN.SEM_WINDOW_SIZE = 7
    cfg.MODEL.SWIN.NUM_SEM_BLOCKS = 1

def add_dinat_config(cfg):
    """
    Add config for NAT Backbone.
    """

    # DINAT transformer backbone
    cfg.MODEL.DiNAT = CN()
    cfg.MODEL.DiNAT.DEPTHS = [3, 4, 18, 5]
    cfg.MODEL.DiNAT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.DiNAT.EMBED_DIM = 64
    cfg.MODEL.DiNAT.MLP_RATIO = 3.0
    cfg.MODEL.DiNAT.NUM_HEADS = [2, 4, 8, 16]
    cfg.MODEL.DiNAT.DROP_PATH_RATE = 0.2
    cfg.MODEL.DiNAT.KERNEL_SIZE = 7
    cfg.MODEL.DiNAT.DILATIONS = [[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]]
    cfg.MODEL.DiNAT.OUT_INDICES = (0, 1, 2, 3)
    cfg.MODEL.DiNAT.QKV_BIAS = True
    cfg.MODEL.DiNAT.QK_SCALE = None
    cfg.MODEL.DiNAT.DROP_RATE = 0
    cfg.MODEL.DiNAT.ATTN_DROP_RATE = 0.
    cfg.MODEL.DiNAT.IN_PATCH_SIZE = 4

def add_convnext_config(cfg):
    """
    Add config for ConvNeXt Backbone.
    """

    # swin transformer backbone
    cfg.MODEL.CONVNEXT = CN()
    cfg.MODEL.CONVNEXT.IN_CHANNELS = 3
    cfg.MODEL.CONVNEXT.DEPTHS = [3, 3, 27, 3]
    cfg.MODEL.CONVNEXT.DIMS = [192, 384, 768, 1536]
    cfg.MODEL.CONVNEXT.DROP_PATH_RATE = 0.4
    cfg.MODEL.CONVNEXT.LSIT = 1.0
    cfg.MODEL.CONVNEXT.OUT_INDICES = [0, 1, 2, 3]
    cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

def add_beit_adapter_config(cfg):
    """
    Add config for BEiT Adapter Backbone.
    """

    # beit adapter backbone
    cfg.MODEL.BEiTAdapter = CN()
    cfg.MODEL.BEiTAdapter.IMG_SIZE = 640
    cfg.MODEL.BEiTAdapter.PATCH_SIZE = 16
    cfg.MODEL.BEiTAdapter.EMBED_DIM = 1024
    cfg.MODEL.BEiTAdapter.DEPTH = 24
    cfg.MODEL.BEiTAdapter.NUM_HEADS = 16
    cfg.MODEL.BEiTAdapter.MLP_RATIO = 4
    cfg.MODEL.BEiTAdapter.QKV_BIAS = True
    cfg.MODEL.BEiTAdapter.USE_ABS_POS_EMB = False
    cfg.MODEL.BEiTAdapter.USE_REL_POS_BIAS = True
    cfg.MODEL.BEiTAdapter.INIT_VALUES = 1e-6
    cfg.MODEL.BEiTAdapter.DROP_PATH_RATE = 0.3
    cfg.MODEL.BEiTAdapter.CONV_INPLANE = 64
    cfg.MODEL.BEiTAdapter.N_POINTS = 4
    cfg.MODEL.BEiTAdapter.DEFORM_NUM_HEADS = 16
    cfg.MODEL.BEiTAdapter.CFFN_RATIO = 0.25
    cfg.MODEL.BEiTAdapter.DEFORM_RATIO = 0.5
    cfg.MODEL.BEiTAdapter.WITH_CP = True
    cfg.MODEL.BEiTAdapter.INTERACTION_INDEXES=[[0, 5], [6, 11], [12, 17], [18, 23]]
    cfg.MODEL.BEiTAdapter.OUT_FEATURES = ["res2", "res3", "res4", "res5"]


def setup_cfg(dataset, model_path, use_swin):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    if use_swin:
      cfg_path = SWIN_CFG_DICT[dataset]
    else:
      cfg_path = DINAT_CFG_DICT[dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.freeze()
    return cfg


class PanopticPrediction:
    """
    Unify different panoptic annotation/prediction formats
    """

    def __init__(self, panoptic_seg, segments_info, metadata=None):
        if segments_info is None:
            assert metadata is not None
            # If "segments_info" is None, we assume "panoptic_img" is a
            # H*W int32 image storing the panoptic_id in the format of
            # category_id * label_divisor + instance_id. We reserve -1 for
            # VOID label.
            label_divisor = metadata.label_divisor
            segments_info = []
            for panoptic_label in np.unique(panoptic_seg.numpy()):
                if panoptic_label == -1:
                    # VOID region.
                    continue
                pred_class = panoptic_label // label_divisor
                isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
                segments_info.append(
                    {
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                        "isthing": bool(isthing),
                    }
                )
        del metadata

        self._seg = panoptic_seg
        self._sinfo = {s["id"]: s for s in segments_info}  # seg id -> seg info
        segment_ids, areas = torch.unique(panoptic_seg.cpu(), sorted=True, return_counts=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Additional Info when using cuda
        if device.type == 'cuda':
            areas = areas.cpu().data.numpy()
        else:
            areas = areas.numpy()
        sorted_idxs = np.argsort(-areas)
        self._seg_ids, self._seg_areas = segment_ids[sorted_idxs], areas[sorted_idxs]
        self._seg_ids = self._seg_ids.tolist()
        for sid, area in zip(self._seg_ids, self._seg_areas):
            if sid in self._sinfo:
                self._sinfo[sid]["area"] = float(area)

    def non_empty_mask(self):
        """
        Returns:
            (H, W) array, a mask for all pixels that have a prediction
        """
        empty_ids = []
        for id in self._seg_ids:
            if id not in self._sinfo:
                empty_ids.append(id)
        if len(empty_ids) == 0:
            return np.zeros(self._seg.shape, dtype=np.uint8)
        assert (
            len(empty_ids) == 1
        ), ">1 ids corresponds to no labels. This is currently not supported"
        return (self._seg != empty_ids[0]).numpy().astype(np.bool)

    def semantic_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)
            if sinfo is None or sinfo["isthing"]:
                # Some pixels (e.g. id 0 in PanopticFPN) have no instance or semantic predictions.
                continue
            yield (self._seg == sid).numpy().astype(np.bool), sinfo

    def instance_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)
            if sinfo is None or not sinfo["isthing"]:
                continue
            mask = (self._seg == sid).numpy().astype(np.bool)
            if mask.sum() > 0:
                yield mask, sinfo



def preprocess(image):
    image = np.asarray(image)
    height, width = image.shape[:2]
    ts_inter1 = datetime.utcnow()

    print("  > Preprocessing image...")
    aug = T.ResizeShortestEdge(
        [640, 640], 2560
    )
    image = aug.get_transform(image).apply_image(image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width, "task": "panoptic"}]
    return inputs

def postprocess(og_image, predictions, stuff_mapper, metadata):
    print("  > Segmenting image...")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    pred = PanopticPrediction(panoptic_seg.cpu(), segments_info, metadata)
    panoptic_preds = list(pred.semantic_masks())

    valid_segments = []
    for segment, sinfo in panoptic_preds:
        if stuff_mapper[sinfo["category_id"]] in exceptions:
            print(f"   >Ignoring stuff --- {stuff_mapper[sinfo['category_id']]}")
            continue
        print(stuff_mapper[sinfo["category_id"]])
        print("-----------------------------------")
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

    wip_img = Image.new(og_image.mode,  og_image.size, "#f2f2f2")
    img1 = Image.composite(og_image, wip_img, mask=bg_blur2)

    print("  > Saving image to local disk...")
    buffered = BytesIO()
    img1.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return {"output_image": img_str}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = torch.load("./image-bg-removal.pkl")
        print("Model loaded...")
        cfg = setup_cfg(dataset, model_path, True)
        self.metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
            )
        if not self.metadata.stuff_classes:
            return {"error": "model could not be loaded"}
        self.stuff_mapper = dict(zip(range(0, len(self.metadata.stuff_classes)), self.metadata.stuff_classes))

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        im = Image.open(image)
        processed_input = preprocess(im)
        output = self.model(processed_input)[0]
        return postprocess(image, predictions=output, stuff_mapper=self.stuff_mapper, metadata=self.metadata)
