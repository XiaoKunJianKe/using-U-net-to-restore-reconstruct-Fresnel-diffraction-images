# using U-net to reconstruct Fresnel diffraction images

## Learning to construct images by Fresnel diffraction
                                    

### *Introduction*

`In`optics, the Fresnel diffraction equation for near-field diffraction approximation of the Kirchhoff-Fresnel diffraction that can be applied to the propagation of waves in the near field. It is used to calculate the diffraction pattern created by waves passing through an aperture or around an object, when viewed from relatively close to the object.
----                                                   
`In` order to demonstrate Fresnel diffraction theory for images (optical wave), I did its simulation and modeling by MATLAB. As shown in figure 1 below, we found that the image [Fig.1 (a)] by Fresnel diffraction has changed [Fig.1 (b)] in content presenting a girl wearing a hat , which one cannot recognize . The research value of the theory such as Fresnel diffraction is significant because its influence mechanism for images is similar foggy weather. Therefore, I first study the Fresnel diffraction theory for images. And the all Fresnel diffraction images are from MATLAB simulation in the rest of the article.
----
![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/linda.png)

Fig. 1. Image(a) is original and image(b) by Fresnel diffraction.
                             
### *Network architecture U-net*

`I` built a U-net CNN with 18 hidden layers that was used to reconstruct the original images from Fresnel diffraction images. Figure 2 shows the details of U-net. This nearly symmetric network architecture comprises a convolutional encoding front end downsampling to capture context and a deconvolutional decoding back end with upsampling for localization Skip connections copy feature layers produced in the contracting path with features layers in the expanding path of the same size, thus improving localization.
----
### *Database*

`I` selected the MNIST database of handwritten digits. I used 5k handwritten digits including number 0 to 9, there are 500 training samples for each number, to train and assess above U-net convolutional neural networks.
----
  ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/network.png)
               
Fig. 2. Details of the implemented U-net type image reconstruction convolutional neural networks.
### *Result*
`The` U-net was implemented using the Tensorflow 1.11.0 Python library on a single NVIDIA GeForce 960M graphics processing unit. Figure 3 presents the results of the reconstruction for Fresnel diffraction images. And the results support a conclusion that the recovery of the original images is possible with Fresnel diffraction images using the U-net CNN.
---
 ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/1.png)
 ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/2.png)
 ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/3.png)
 ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/4.png)
 ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/5.png)
 ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/6.png)
 ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/7.png)
 ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/8.png)
 ![](https://github.com/XiaoKunJianKe/using-U-net-to-restore-reconstruct-Fresnel-diffraction-images/blob/master/images/9.png)
Fig. 3. Fresnel diffraction_image: Fresnel diffraction image simulated by matlab; reconstruction_label: reconstructed images from  Fresnel diffraction image by U-net; original_babel: original image; binarization_label: The image is binarized to gray 0 and 1.

### *Appendix about codes* 
#### MATLAB
##### convert_jpg、fun_mnist: obtain original mnist database and Fresnel diffraction image database.
##### mnist_all: original digit database. and style： .mat
#### Python
##### mode: U-net architecture;
##### train_unet_mnist: train mode;
##### te_st: test trained networks;
##### log: trained mode parameters.



                         
