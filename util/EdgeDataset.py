import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import os

class SRDataset(Dataset):
    """
        SRDataset: data set for loading images in a single directory
            with no labels
            
        An EdgeDataset is stored as a directory that has the following
            components:
            
            /files.csv -- contains a list of file names, without directory path
            /img/ -- the directory where the original images are stored
            /hr/ -- the HR images, used as input
            /lr2x/ , /lr4x/ ... -- the LR images
            /edge/ -- the edge generated from HR images
            
            Note: These images should be generated beforehand.
        Each line in files.csv correspond to a file in `img/`
    """
    def __init__(self, img_dir, img_type="img", augment=False):
        """
            img_dir: the directory of images
            img_type: which type of image to use, 
                can be "img", "hr", "lr2x", "lr4x", "edge" ...
                default is "img"
        """
        super().__init__()
        self.img_dir = img_dir
        self.img_list = pd.read_csv(os.path.join(img_dir, "files.csv"),
                                    names=["filename"])
        self.img_type = type
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, "img", self.img_list.iloc[idx].filename)   
        image = read_image(img_path)
        return image

    
    def __len__(self):
        return len(self.img_list)