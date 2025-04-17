import torch
import json
import os
from PIL import Image

class HatefulMemesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, img_only=True, limit_to_examples=None, transform=None):
        self.data = [json.loads(l) for l in open(data_path)]
        if limit_to_examples & limit_to_examples > 0:
            self.data = self.data[:limit_to_examples]
        self.data_dir = os.path.dirname(data_path)
        self.img_only = img_only
        self.transform = transform
            
    def __getitem__(self, index: int):
        # Load images on the fly.
        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = self.data[index]["text"]
        label = self.data[index]["label"]
        if self.img_only:
            return image, label
        else:
            return image, text, label

    def load_image_only(self, index: int):
        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        return image
    
    def get_label(self, index: int):
        label = self.data[index]["label"]
        return label
    
    def __len__(self):
        return len(self.data)