# CS766 Project
## Edge-based Image Super-Resolution
##### Group members: Yunhan Hu, Ruohui Wang, Xin Yuan



### Single Image Super-Resolution

Image super-resolution is the task of generating high-resolution images from low-resolution images. Due to the limitations of some actual scenes, only one low-resolution image is often available. Super-resolution using only one low-resolution image is single-image super-resolution. Since one low-resolution image corresponds to several high-resolution images, super-resolution is an ill-defined problem. Therefore, additional prior knowledge is needed to constrain the recovered high-frequency information. The state-of-the-art methods are based on deep learning. Specifically, it uses the similarity of different images in high-frequency details to obtain the relationship between high-resolution and low-resolution images to guide the reconstruction of high-resolution images.

![Low resolution](2021-05-04-22-29-22.png)$\Rightarrow$![](2021-05-04-22-30-15.png)

### Applications

In addition to being a research direction in computer vision, super-resolution also has many practical applications. For example, video enhancement can repair some old videos. Various medical imaging is limited by physical conditions and radiation safety, and the resolution cannot be increased arbitrarily. In this case, super-resolution can improve the quality of medical images. Similarly, remote sensing and astronomical observations are limited by the optical diffraction limit, and the resolution is difficult to improve, so super-resolution can also be used.


- Regular video information enhancement
- Surveillance
- Medical diagnosis
- Earth-observation remote sensing
- Astronomical observation
- Biometric information identification


### Approach

In this project, we proposed a two-stage generative adversarial network (GAN) for single image super-resolution. The first stage is the high-resolution edge prediction network (G1), which takes in the low-resolution image and edge as the input, outputs the predicted high-resolution edge image. The second stage is the image inpainting network (G2), which takes in the predicted high-resolution edge from G1 and low resolution image, outputs the final super-resolution image.

![](2021-05-05-00-28-02.png)

### Implementation

#### Dataset

The superresolution task requires many types of the original images -- a single image needs to be scaled to 512x512 to be processed by the network, and aside from that, it has multiple downscaled version (2x, 4x for example). In edge-informed super resolution, we also need the detected edges of these images, either by Canny or the edge model. To manage images of these many type, and choose the versions of the images we want, we implemented `SRDataset`, a class that can generate, feed images. This improves the performance compared to the authors' version that preprocesses the image during training.