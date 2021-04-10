import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import os
from torchvision.io import write_jpeg

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
            /edge_lr2x/,/edge_lr4x/,/edge_lr8x/ LR edge from LR images

            Note: These images should be generated before use. Check `generate_image()` for details
        Each line in files.csv correspond to a file in `img/`
    """

    def __init__(self, img_dir, img_type="img", img_list="files.csv", augment=False):
        """
            img_dir: the directory of images
            img_type: which type of image to use, 
                can be "img", "hr", "lr2x", "lr4x", "lr8x", "edge", 
                    or a can be list of them
                    for example, ["hr", "edge"] will give two images
                default is "img"
            img_list: the file is contains the list of pictures used, 
                expressed as rows in a csv table, by default it is "files.csv",
                which contains all files in the directory
        """
        super().__init__()
        self.img_dir = img_dir
        self.img_list = pd.read_csv(os.path.join(img_dir, img_list),
                                    names=["filename"])
        self.img_type = img_type

    def __getitem__(self, idx):
        if type(self.img_type) is list:
            img_paths = []
            for img_type in self.img_type:
                img_path = os.path.join(self.img_dir, img_type,
                                        self.img_list.iloc[idx].filename)
                if not os.path.exists(img_path):
                    self.generate_image(img_type,idx)
                
                img_paths.append(img_path)
            return tuple(read_image(img_path).float()/255 for img_path in img_paths)
        else:
            img_path = os.path.join(self.img_dir, self.img_type,
                                    self.img_list.iloc[idx].filename)
            
            if not os.path.exists(img_path):
                self.generate_image(img_type,idx)
            image = read_image(img_path).float()/255
            return image

    def __len__(self):
        return len(self.img_list)

    def generate_image(self, img_type, idx='all', model=None):
        """
            Generate the images.
            img_type: can be "edge", "hr", "lr2x", "lr4x", "lr8x","edge_lr2x","edge_lr4x","edge_lr8x"
                      can also be "pred_edge_lr2x","pred_edge_lr4x","pred_edge_lr8x"
            Edge can only be generated if the image with corresponding resolution exists

        """
        if idx == 'all':
            idx = self.img_list["filename"]
        else:
            idx=[self.img_list.iloc[idx].filename]

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
            for img_name in idx:
                img_path = os.path.join(self.img_dir, "hr",  img_name)
                img = imread(img_path)

                if img.shape != (size/downscale, size/downscale, 3):
                    img = resize(img, (size/downscale, size /
                                       downscale), anti_aliasing=True)

                img = img_as_ubyte(img)
                hr_path = os.path.join(self.img_dir, img_type, img_name)
                imsave(hr_path, img)

        elif img_type.startswith("edge"):
            # low resolution edge images ?
            if img_type == "edge":
                downscale, edge_src = 1, "hr"
            elif img_type.find("lr2x")+1:
                downscale, edge_src = 2, "lr2x"
            elif img_type.find("lr4x")+1:
                downscale, edge_src = 4, "lr4x"
            elif img_type.find("lr8x")+1:
                downscale, edge_src = 8, "lr8x"
            else:
                raise NotImplementedError

            from skimage.feature import canny
            from skimage.color import rgb2gray
            os.makedirs(os.path.join(self.img_dir, img_type), exist_ok=True)
            for img_name in idx:
                img_path = os.path.join(self.img_dir, edge_src, img_name)
                img = imread(img_path)

                edge_img = canny(rgb2gray(img), sigma=2.0)
                edge_img = img_as_ubyte(edge_img)
                edge_path = os.path.join(self.img_dir, img_type, img_name)
                imsave(edge_path, edge_img)

        elif img_type == "hr":
            os.makedirs(os.path.join(self.img_dir, img_type), exist_ok=True)
            for img_name in idx:
                img_path = os.path.join(self.img_dir, "img", img_name)
                img = imread(img_path)

                # TODO: use crop as in the paper
                if img.shape != (size, size, 3):
                    img = resize(img, (size, size), anti_aliasing=True)

                img = img_as_ubyte(img)
                hr_path = os.path.join(self.img_dir, "hr", img_name)
                imsave(hr_path, img)

        elif img_type.startswith("pred_edge"):
            if model.config.SCALE != downscale:
                print("Please check if the model is using the same scale factor!")
                raise NotImplementedError

            if img_type == "pred_edge_lr2x":
                downscale = 2
            elif img_type == "pred_edge_lr4x":
                downscale = 4
            elif img_type == "pred_edge_lr8x":
                downscale = 8
            else:
                raise NotImplementedError
            
            os.makedirs(os.path.join(self.img_dir, img_type), exist_ok=True)
            device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu") 
            
            for img_name in idx:
                img_path = os.path.join(self.img_dir, "lr"+str(downscale)+"x", img_name)
                lr_img = (read_image(img_path).float()/255).to(device)
                img_path = os.path.join(self.img_dir, "edge_lr"+str(downscale)+"x", img_name)
                lr_edge = (read_image(img_path).float()/255).to(device)
                
                pred_edge=model(lr_img.unsqueeze_(0),lr_edge.unsqueeze_(0))
                edge_path = os.path.join(self.img_dir, img_type, img_name)
                imsave_tensor(pred_edge[0,:,:,:], edge_path)

def imsave_tensor(imgtensor,path):
    """
    Input a CHW tensor in float between 0 and 1
    Save into *.jpg file
    """
    if imgtensor.device.type != 'cpu':
        imgtensor = imgtensor.to('cpu')
    imgtensor = imgtensor * 255
    imgtensor = imgtensor.byte()
    write_jpeg(imgtensor,path)