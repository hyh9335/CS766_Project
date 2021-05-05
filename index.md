## Architecture

The network has two components, the edge detection model and the super-resolution model.

Both the model use a similar architecture.

## Implementation

### Dataset

The superresolution task requires many types of the original images -- a single image needs to be scaled to 512x512 to be processed by the network, and aside from that, it has multiple downscaled version (2x, 4x for example). In edge-informed super resolution, we also need the detected edges of these images, either by Canny or the edge model. To manage images of these many type, and choose the versions of the images we want, we implemented `SRDataset`, a class that can generate, feed images. This improves the performance compared to the authors' version that preprocesses the image during training.

## Result