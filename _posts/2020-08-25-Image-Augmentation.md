---
layout: post
title: "How you can do Image Data Augmentation using Keras"
date: 2020-08-25
---

![](/images/john-price-wzRQfEw9CMc-unsplash.jpg)

This summer I am doing my dissertation on Transfer learning in the field of multi-spectral satellite images. When I started off on it by collecting relevant dataset, I realised that the data was not large enough for my deep learning models to train on and give good prediction scores. For this purpose, I decided to explore Image Augmentation to increase my data size. After doing couple of MOOCs aroud Image Augmentation using Keras, I realised that I was still strugging a bit in implementing the process in my own project. So I decided to this articel as a one-stop solution for any one who is in the same situation and is looking for a straight-forward solution with full and easy to understand explanations. 
In this article, you wil find the following details -
- What is augmentation and why it is needed, 
- The Keras code to implement augmentation, and
- Useful links for your further reference

#### What is Image Augmentation and why it is needed
Image Augmentation is a technique of artificially increasing the size of the image data without acquiring new images. Images are duplicated with some kind of variation so that the data size can increase and model can learn from more examples or bigger data. Idea is to rearrange the pixels in a way that this adds enough noise without disturbing the original features. This way the original labels are preserved. There are different ways to add noise or to increase data size such as random rotation, shifts, shear, flips etc. These increased images can improve the generalizability of a model.

Image Augmentation is needed because in real-world use cases, you will have small labelled datasets to train your model on. Although, Convolutional Neural Nets are highly efficient networks and can learn very good amount of features from even small image datasets. Augmentation provides a more comprehensive set of possible data points or features so that training and validation loss can both be minimized. In this way, Image Augmentation solves the problem of overfitting by handling the training dataset itself. In addition to availability of only small dataset, there is a problem of class imbabalance in data that is available, meaning a varying amount of data is present for different classes. This can skew the model in favor of one class after training. This problem of class imbalance can also be addressed by augmenting the data for the classes with lesser number of images.

A generalizable image analysis model must be invarient to changes in viewpoint, background, scale, orientation etc. Augmentation processes take care of this requirement by doing operations like flipping, shifting, rotating, and many more. The famous AlexNet in 2012 by Alex Krizhevsky revolutionized image classification by using CNNs over the Imagenet dataset. He used image augmentation techniques like flipping, cropping patches and changing color intensity using PCA to increase the data size by a magnitude of 2048. They reported reducing overfitting and thus the test error rates.

Offlate, researchers have introduced a variety of new neural aprroaches of image augmentation,, namely - Neural Augmentation, Auto Augment, Smart Augmentation and GAN based Data Augmentation. Scope of this article is only limited to translational methods of image augmentation and will be coveering nural methods in detail in another article. 

An example of image augmentation on a 64x64 RGB or colored image -

![](/images/_2_1839230.png)     ![](/images/_2_2989151.png)     ![](/images/_2_5985086.png)     ![](/images/_2_6416129.png) 

#### The Keras code to implement augmentation
Please find the code on my github repository at this [link](https://github.com/yvrjsharma/Image-Analysis/blob/master/Image%20Augmentation%20Using%20Keras%20ImageDataGenerator.ipynb). Please note that it is a Jupyter Notebook along with comments to better unerstand the flow.

#### Useful links for your further reference
- [Image Augmentation for Convolutional Neural Networks](https://opendatascience.com/image-augmentation-for-convolutional-neural-networks/)
- [Image Augmentation for Deep Learning](https://towardsdatascience.com/image-augmentation-for-deep-learning-histogram-equalization-a71387f609b2)
- [Keras Official Blog: Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- Shorten C, Khoshgoftaar TM (2019) A survey on image data augmentation for deep learning. Journal of Big Data 6(1), DOI 10.1186/s40537-019-0197-0
- Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks. Adv Neural Inf Process Syst. 2012;25:1106–14.

