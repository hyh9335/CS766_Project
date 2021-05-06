import os
#import yaml

DEFAULT_CONFIG = {
    'DATAPATH': ['dataset','Set14'],  # directory of dataset in the list form
    'MODEL_PATH': ['models'],       # directory of the trained model
    
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'MODEL': 1,                     # 1: edge model, 2: SR model, 3: SR model with edge enhancer
    'SCALE': 2,                  # scale factor (2, 4, 8)
    'SEED': 10,                     # random seed
    'GPU': [0],                     # list of gpu ids
    'DEBUG': 0,                     # turns on debugging mode
    'VERBOSE': 0,                   # turns on verbose mode in the output console

    'LR': 0.0001,                # learning rate
    'BETA1': 0.0,                # adam optimizer beta1
    'BETA2': 0.9,                # adam optimizer beta2
    'BATCH_SIZE': 8,                # input batch size for trainingceleba-hq
    'HR_SIZE': 256,                 # HR image size for training 0 for original size
    'SIGMA': 2,                     # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
    'MAX_ITERS': 2e7,               # maximum number of iterations to train the model
    'EDGE_THRESHOLD': 0.5,          # edge detection threshold

    'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    'FM_LOSS_WEIGHT': 10,           # feature-matching loss weight
    'STYLE_LOSS_WEIGHT': 250,       # style loss weight
    'CONTENT_LOSS_WEIGHT': 0.1,     # content loss weight
    'ADV_LOSS_WEIGHT1': 0.1,        # edge model adversarial loss weight
    'ADV_LOSS_WEIGHT2': 1,          # SR model adversarial loss weight
    'GAN_LOSS': 'hinge',         # nsgan | lsgan | hinge

    'SAVE_INTERVAL': 1000,          # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 1000,        # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,              # number of images to sample
    'EVAL_INTERVAL': 0,             # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 10,             # how many iterations to wait before logging training status (0: never)
}

class Config(dict):
    def __init__(self, copy=DEFAULT_CONFIG):
        super().__init__()
        if copy is not None:
            for k in copy:
                self[k] = copy[k]
    
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            return None
    
    def __setattr__(self, key, value):
        self[key] = value

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        for k in self:
            print(f'{k} : {self[k]}')
        print('')
        print('---------------------------------')
        print('')


