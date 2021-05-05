# CS766 Project
## Edge-based Image Super-Resolution
##### Group members: Yunhan Hu, Ruohui Wang, Xin Yuan



### Single Image Super-Resolution

Image super-resolution is the task of generating high-resolution images from low-resolution images. Due to the limitations of some actual scenes, only one low-resolution image is often available. Super-resolution using only one low-resolution image is single-image super-resolution. Since one low-resolution image corresponds to several high-resolution images, super-resolution is an ill-defined problem. Therefore, additional prior knowledge is needed to constrain the recovered high-frequency information. The state-of-the-art methods are based on deep learning. Specifically, it uses the similarity of different images in high-frequency details to obtain the relationship between high-resolution and low-resolution images to guide the reconstruction of high-resolution images.

![Low resolution](2021-05-04-22-29-22.png)--->![High resolution](2021-05-04-22-30-15.png)

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

#### Source codes
The source codes are at [CS766_Project](https://github.com/hyh9335/CS766_Project).

### Result and comparsion
#### Result
The super-resolution results of SET14 are in [2x downsampling](https://github.com/hyh9335/CS766_Project/tree/gh-pages/pred_full_lr2x) and [4x downsampling](https://github.com/hyh9335/CS766_Project/tree/gh-pages/pred_full_lr4x).

#### Comparsion

| Model | PSNR | SSIM |
|----|----|----|
|Bicubic|25.99|0.7031|
|SRCNN|27.50|0.7513|
|SRGAN|26.02|0.7397|
|Ours|25.19|0.9134|

This is the result of our network evaluation. The testing set is set 14. Downsampling is 4 times. The results show that the PSNR is low. This is because the loss function of our network is not PSNR-oriented, and the network has not fully converged due to limited computing power. SSIM is very high. This is normal because we use edge images, which is equivalent to inputting structural information into the network.

![](2021-05-05-17-49-11.png)
These images are ground truth, bicubic, SRCNN, SRGAN and our network. As a result, our image is better than bicubic and SRCNN, and is close to the result of SRGAN. However, our pictures have chromatic aberration and noise, which is why the PSNR is low.

### Problems
![](img_001_SRF_4_HR.png)

The picture above is the output of our network. As you can see, there are white patches in the picture. This phenomenon sometimes appears in the output results. There are several possible reasons for this phenomenon. One is the instability of GAN training. How to train GAN has always been the difficulty of GAN. The second is that insufficient computing power has caused the network to not fully converge. The third is the problem of the network structure itself. A similar phenomenon appeared in the original paper.
The solution is to replace the GAN network structure and try to use a deeper network for training, such as SRGAN.