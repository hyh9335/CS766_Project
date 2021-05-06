# CS766_Project

## Requirements

pytorch >= 1.8.0 with cuda
pytorch_msssim
pandas
scikit-image

## How to use

1. Prepare the SR data set
    From `SRDataset.py` documentation:
        An SRDataset is a directory that includes the following components:

            /files.csv -- contains a list of file names, without directory path
            /img/ -- the directory where the original images are stored
            /hr/ -- the HR images, used as input, they are scaled and cropped from the original images

            /lr2x/, /lr4x/, /lr8x/ -- the LR images
            /edge/ -- the edge generated from HR images
            /edge_lr2x/,/edge_lr4x/,/edge_lr8x/ LR edge from LR images

            /pred_edge_lr2x/, etc. -- predicted edges
            /pred_full_lr2x/, etc. -- the SR images

            Note: Some images should be generated before use.
        Each line in files.csv correspond to a file in `img/`

2. Specify the specs: the path to the data, the path of the trained model etc.
    * Refering to config.json for more information.
3. Run `python3 main.py [config file name]`
    * By default config.json is used