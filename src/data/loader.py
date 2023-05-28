from torch.utils.data import Dataset
from skimage import io
import os
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
from dataframe import make_csv

class CancerDataLoader(Dataset):
    def __init__(self, img_root=None, csv_root=None, transform=None) -> None:
            super().__init__()
            self.annotations = make_csv(csv_root)
            self.root_dir = img_root
            self.transform = transform

    def __len__(self):
        #return int(len(self.annotations)*0.05)
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        label = torch.tensor(int(self.annotations.iloc[index,1]))
        name = self.annotations.iloc[index, 0]
        if self.transform:
            image = self.transform(image)
            
        return (image, label, name)