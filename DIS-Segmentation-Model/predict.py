from typing import List
from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from isnet.model import ISNetDIS
from io import BytesIO

# project imports
from isnet.data_loader_cache import normalize, im_reader, im_preprocess
from PIL import Image, ImageFilter
import pillow_avif
from pillow_heif import register_heif_opener

register_heif_opener()

config = {}
config["model_path"] ="./saved_models"
config["restore_model"] = "isnet.pth"
config["interm_sup"] = False
config["model_digit"] = "full"
config["seed"] = 0
config["device"] = "cuda"
config["cache_size"] = [1024, 1024]
config["input_size"] = [1024, 1024]
config["crop_size"] = [1024, 1024]
config["model"] = ISNetDIS()
cpu_device = torch.device("cpu")


class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image


transform =  transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])

def load_image(img):
    im, im_shp = im_preprocess(np.array(img), config["cache_size"])
    im = torch.divide(im,255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)

def run_inference(net,  inputs_val, shapes_val):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if(config["model_digit"]=="full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(config["device"])
    ds_val = net(inputs_val_v)[0]
    pred_val = ds_val[0][0,:,:,:]
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[0][0],shapes_val[0][1]),mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi) # max = 1

    if config["device"] == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8)

class Predictor(BasePredictor):
    def setup(self):
        model = config["model"]
        # convert to half precision
        if(config["model_digit"]=="half"):
            model.half()
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()

        model.to(config["device"])

        if(config["restore_model"]!=""):
            model.load_state_dict(torch.load(config["model_path"]+"/"+config["restore_model"],map_location=config["device"]))
            model.to(config["device"])
        model.eval()
        self.predictor = model
        print("Model loaded...")

    def predict(
        self,
        image: Path = Input(description="Upload image in following formats: jpg, jpeg, png, heic, webp, heif, avif"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        img = Image.open(image).convert("RGB")
        if img.format in ["AVIF", "HEIC"]:
            img = img.convert("RGB")
        image_tensor, orig_size = load_image(img)
        background = run_inference(net=self.predictor, inputs_val=image_tensor, shapes_val=orig_size)
        background = Image.fromarray(background)
        background.save("background.png")
        bg_blur = background.convert("L")
        bg_blur2 = bg_blur.filter(ImageFilter.BoxBlur(1))
        wip_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        final_img = Image.composite(img, wip_img, mask=bg_blur2)
        final_img.save("output.png")
        return [Path("output.png"), Path("background.png")] # type: ignore
