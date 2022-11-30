from cog import BasePredictor, Input, Path
import torch
from detectron2.data import MetadataCatalog
from PIL import Image
import detectron2.data.transforms as T
import numpy as np
from datetime import datetime
import pickle

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
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = torch.load("./image-bg-removal.pkl")
        print("Model loaded...")

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        im = Image.open(image)
        processed_input = preprocess(im)
        output = self.model(processed_input)[0]
        print(output["panoptic_seg"])
        with open("output_dump.pkl", "wb") as pkl_dump:
            pickle.dump(output["panoptic_seg"], pkl_dump)
        return "output_dump.pkl"
