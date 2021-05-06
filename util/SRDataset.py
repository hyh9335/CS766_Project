import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, write_jpeg
import pandas as pd
import os
from torchvision.io.image import ImageReadMode

class SRDataset(Dataset):
    """
        SRDataset: data set for loading images in a single directory
            with no labels

        An SRDataset is stored as a directory that has the following
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
        self.mode = ImageReadMode
        self.threshold = 0.5
        

    def __getitem__(self, idx):
        if type(self.img_type) is list:
            img_data = []
            for img_type in self.img_type:
                img_path = os.path.join(self.img_dir, img_type,
                                        self.img_list.iloc[idx].filename)
                if not os.path.exists(img_path):
                    self.generate_image(img_type,idx)
            
                m = self.mode.GRAY if img_type.find('edge') + 1 else self.mode.RGB

                image = read_image(img_path, mode = m).float()/255
                img_data.append(image)

            return tuple(image for image in img_data)
        else:
            img_path = os.path.join(self.img_dir, self.img_type,
                                    self.img_list.iloc[idx].filename)
            
            if not os.path.exists(img_path):
                self.generate_image(img_type,idx)
            
            m = self.mode.GRAY if self.img_type.find('edge') + 1 else self.mode.RGB

            image = read_image(img_path, mode = m).float()/255
            return image

    def __len__(self):
        return len(self.img_list)

    def show(self,idx,width = 512, height = 512):
        from PIL import Image
        img_path = [os.path.join(self.img_dir, img_type,
                                        self.img_list.iloc[idx].filename) for img_type in self.img_type]
        img_list = [Image.open(path).resize((width, height), Image.BILINEAR) for path in img_path]
        result = Image.new(img_list[0].mode, (width * len(img_list), height))
        for i, im in enumerate(img_list):
            result.paste(im, box=(i * width, 0))
        result.save(os.path.join(self.img_dir, self.img_list.iloc[idx].filename))

    def generate_edge(self, edge_src, idx, tensigma=20):
        from skimage.feature import canny
        from skimage.color import rgb2gray
        from skimage.io import imread, imsave
        from skimage import img_as_ubyte

        idx = [self.img_list.iloc[idx].filename]
        os.makedirs(os.path.join(self.img_dir, 'canny_edge', str(tensigma)), exist_ok=True)
        
        for img_name in idx:
            img_path = os.path.join(self.img_dir, edge_src, img_name)
            img = imread(img_path)

            edge_img = canny(rgb2gray(img), sigma=tensigma/10)
            edge_img = img_as_ubyte(edge_img)
            edge_path = os.path.join(self.img_dir, 'canny_edge', str(tensigma),img_name)
            imsave(edge_path, edge_img)

    def generate_image(self, img_type, idx='all', model=None , no_single_point=True):
        """
            Generate the images.
            img_type: can be "edge", "hr", "lr2x", "lr4x", "lr8x","edge_lr2x","edge_lr4x","edge_lr8x"
                      can also be "pred_edge_lr2x","pred_edge_lr4x","pred_edge_lr8x"
                      can also be "sr_bicubic_2x", "edge_sr_bicubic_2x", etc
            Edge can only be generated if the image with corresponding resolution exists

        """
        if idx == 'all':
            idx = self.img_list["filename"]
        else:
            idx = [self.img_list.iloc[idx].filename]
            
        from skimage.transform import resize, rescale
        from skimage.io import imread, imsave
        from skimage import img_as_ubyte

        size = 512

        if img_type.startswith("sr_bicubic"):
            if img_type == "sr_bicubic_2x":
                downscale, img_src = 2, "lr2x"
            elif img_type == "sr_bicubic_4x":
                downscale, img_src = 4, "lr4x"
            elif img_type == "sr_bicubic_8x":
                downscale, img_src = 8, "lr8x"
            else:
                raise NotImplementedError

            os.makedirs(os.path.join(self.img_dir, img_type), exist_ok=True)
            for img_name in idx:
                img_path = os.path.join(self.img_dir, img_src,  img_name)
                img = imread(img_path)

                if img.shape != (size, size, 3):
                    img = resize(img, (size, size), anti_aliasing=True, order=3)

                img = img_as_ubyte(img)
                sr_path = os.path.join(self.img_dir, img_type, img_name)
                imsave(sr_path, img)


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
                    img = center_crop_resize(img, size/downscale)

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
            elif img_type.find("sr_bicubic")+1:
                edge_src = img_type[5:]
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
                    img = center_crop_resize(img, size)

                img = img_as_ubyte(img)
                hr_path = os.path.join(self.img_dir, "hr", img_name)
                imsave(hr_path, img)

        elif img_type.startswith("pred"):
            _, model_type, gen_img_type = img_type.split('_')

            import re
            downscale = re.search(r'''(?P<scale>\d+)x''', gen_img_type).group('scale')
            downscale = int(downscale)
            
            if model.config.SCALE != downscale:
                print("Please check if the model is using the same scale factor!")
                raise NotImplementedError
            
            os.makedirs(os.path.join(self.img_dir, img_type), exist_ok=True)
            device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu") 
            
            for img_name in idx:
                img_path = os.path.join(self.img_dir, "lr{}x".format(downscale), img_name)
                lr_img = (read_image(img_path,mode = self.mode.RGB).float()/255).to(device)
                
                img_path = os.path.join(self.img_dir, "edge_lr{}x".format(downscale), img_name)
                lr_edge = (read_image(img_path).float()/255).to(device)

                with torch.no_grad():
                    pred_img = model(lr_img.unsqueeze_(0), lr_edge.unsqueeze_(0))
                
                if model_type == 'edge':
                    outputs = (pred_img > self.threshold)
                    if no_single_point == True:
                        filters = torch.ones(1, 1, 3, 3).to(device)
                        nearby = torch.nn.functional.conv2d(outputs.float(), filters.float(), padding=1)
                        nearby =(nearby > torch.tensor(1.5))
                        outputs = outputs & nearby
                elif model_type == 'full':
                    outputs = pred_img

                output_path = os.path.join(self.img_dir, img_type, img_name)
                imsave_tensor(outputs[0,:,:,:], output_path)

def imsave_tensor(imgtensor, path):
    """
    Input a CHW tensor in float between 0 and 1
    Save into *.jpg file
    """
    if imgtensor.device.type != 'cpu':
        imgtensor = imgtensor.to('cpu')
    imgtensor = imgtensor * 255
    imgtensor = imgtensor.byte()
    write_jpeg(imgtensor,path)


def center_crop_resize(img, size):
    """
    crop a square as large as possible from the center of `img`,
        and resize it to `size`
    """
    from numpy import minimum
    from skimage.transform import resize

    imgh, imgw = img.shape[0:2]

    if imgh != imgw:
        # center crop, from knazeri/edege-informed-sisr
        side = minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]
    
    return resize(img, (size, size), anti_aliasing=True)