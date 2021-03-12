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
            /lr2x/ , /lr4x/, /lr8x/ -- the LR images
            /edge/ -- the edge generated from HR images
            
            Note: These images should be generated before use. Check `generate_image()` for details
        Each line in files.csv correspond to a file in `img/`
    """
    def __init__(self, img_dir, img_type="img", augment=False):
        """
            img_dir: the directory of images
            img_type: which type of image to use, 
                can be "img", "hr", "lr2x", "lr4x", "lr8x", "edge", 
                    or a can be list of them
                    for example, ["hr", "edge"] will give two images
                default is "img"
        """
        super().__init__()
        self.img_dir = img_dir
        self.img_list = pd.read_csv(os.path.join(img_dir, "files.csv"),
                                    names=["filename"])
        self.img_type = img_type
        
    def __getitem__(self, idx):
        if type(self.img_type) is list:
            img_paths = []
            for img_type in self.img_type:
                img_path = os.path.join(self.img_dir, img_type, 
                    self.img_list.iloc[idx].filename) 
                img_paths.append(img_path)
            return tuple(read_image(img_path) for img_path in img_paths)
        else:
            img_path = os.path.join(self.img_dir, self.img_type, 
                        self.img_list.iloc[idx].filename)
            image = read_image(img_path)
            return image

    def __len__(self):
        return len(self.img_list)

    def generate_image(self, img_type):
        """
            Generate the images.
            img_type: can be "edge", "hr", "lr2x", "lr4x", "lr8x"
        """
        

        from skimage.transform import resize, rescale
        from skimage.io import imread, imsave
        from skimage import img_as_ubyte

        size = 512
        if img_type.startswith("lr"):
            if img_type == "lr2x":
                downscale = 2
            elif img_type == "lr4x":
                downscale = 4
            elif img_type == "lr8x":
                downscale = 8
            else:
                raise NotImplementedError
            os.makedirs(os.path.join(self.img_dir, img_type), exist_ok=True)
            raise NotImplementedError

        elif img_type == "edge":
            from skimage.feature import canny
            from skimage.color import rgb2gray
            os.makedirs(os.path.join(self.img_dir, img_type), exist_ok=True)
            for img_name in self.img_list["filename"]:
                img_path = os.path.join(self.img_dir, "img", img_name)
                img = imread(img_path)

                if img.shape != (size, size, 3):
                    img = resize(img, (size, size), anti_aliasing=True)

                edge_img = canny(rgb2gray(img), sigma=2.0)
                edge_img = img_as_ubyte(edge_img)
                edge_path = os.path.join(self.img_dir, "edge", img_name)
                imsave(edge_path, edge_img)

        elif img_type == "hr":
            os.makedirs(os.path.join(self.img_dir, img_type), exist_ok=True)
            for img_name in self.img_list["filename"]:
                img_path = os.path.join(self.img_dir, "img", img_name)
                img = imread(img_path)

                if img.shape != (size, size, 3):
                    img = resize(img, (size, size), anti_aliasing=True)
                
                img = img_as_ubyte(img)
                hr_path = os.path.join(self.img_dir, "hr", img_name)
                imsave(hr_path, img)