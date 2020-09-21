---
layout: post
title: "Gathering Data, Building a Computer Vision model, and Deploying the model"
date: 2020-09-17
---
![](/images/photos-hobby-g29arbbvPjo-unsplash.jpg)

## Why this article

Hi Everyone, so this is my first blog on the [Fastai](https://www.amazon.in/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) book that I have started reading recently. It has been a fascinating read so far. I am have just completed my second chapter, and I have already built a high quality deep learning image classificatoion model. Though, I am not a beginner in this field, and I understand the math and programming architecture very well behind the state of the art networks used in  building the model (Resne18 in this particular case), this chapter basically teaches you how to build a DL model with very few lines of codes and also deploy it as a web application. This last part of deploying the model has what fascinated me the most in this chapter. So, I figured that most of you might not be interested in putting hours behind this book, due to your already busy schedules, it will be beneficial for you to read this blog and use this to build your first model along with deploying it on the web. Jeremy and Sylvain's book has a dedicated website too on which you can simply go and read all the cahpters and copy code to run in your own environment. Still, oit leaves out a lot of work for you to figure out as you go. For example, I encountered a number of errors and confusions while impleneting this particular chapter and would be glad to share my learning through this blog, so that anyone who is not reading the book, or who has read it and is stuck somewhere, can use this articlee to sort them out.

## Gathering Data

![](/images/hector-j-rivas-QNc9tTNHRyI-unsplash.jpg)

I have used Microsoft Bing's limited period image free image download web API. The free API version enables me to download 150 images in one go. For this I had to create an Azure account and register for Bing Image earch application over there, and I will be alking you through my steps and will guide you againts possible hickups that I encountered during my efforts. This was an interetsing and fun part in the entire effort of building a model and then deploying it.
This part is crucial for a data scientist because untill and unless you have clean training and validation data, you cant really build a model. A Data Science project pipleline starts with Gathering and Cleaning of data and ends up at Deploying the model.

## Building a Computer Vision model

![](/images/chap2_fastai.jpeg)

So, I have build a fun web appliction using the techniques that i have learned from this chapter. My web application identifies whether the incoming image (one image at a time) is of the actress Jessica Chastain or Bryce Dallas Howard. So, the idea is to identify an incoming image to onee of the two computer vision classes. Now, I could have gone with classiffying between images of Cats and Dogs or between different food items or between different clothing items, but I chose this particular case of doppelganger as these two actress have confused me previously. I used to get confused betwenn the two them and this has been acknowledged by various online publications as well. Okay, fun apart, this has been an interesting application to develop and even though my model was abel to predict with 91% validation accuracy between the two actresses, new incoming images of Bryce Dallas Howard were often confused with Jessica Chastain, confirming my doubts that sometimes it is [very difficult to identify between them who's who](https://www.today.com/video/can-you-tell-jessica-chastain-and-bryce-dallas-howard-apart-461250627836).

## Deploying the model
![](/images/mybinder.png)

I have used Binder, which is a **free** and _simplest_ method out there to publish your web app online. This allows you to host a POC type or a hobby type project of yours for bigger community to see easily by clicking on a link. I will be sharing below my web application's link as well as how you can do the same. Please keep in mind that this is a free support available online, so it might suffer some glitches or lag time when you will try to access my application online.

## Process in details

This chapter basically allays the doubts like you need vast amount of data and large number of codelines to do deep learning or to build highly acccuracte AI models. This is made possible by using the power of Transfer Learning, Data Augmentation and fastai libraries which are build over Pytorch framework. This is the first time I am using anytihng other then Tensorfloww or Keras, and I must say fastai library is quite intuitive and easy to understand and pick. In any supervised deep learning project, you need good quality data and corresponding labels. Lets start by creating data repositories first.

### 1.  Data Preparation
In this project, I am basically developing an _actress detector_ model. So first of all i need some images of Jessica Chastain and Bryce Dallas Howard. I am using Bing Image Search API for this. As far as I know, this is the best available option out there if you want to build small size datasets for your POCs or hobby projects.It is free for upto 1000 queries per month and allows you to download 150 images at a time. Bing Image Search API is available as a part of Azure Cognitive Services. For this follow below steps in order:

1. Create your free account on Azure [portal](https://azure.microsoft.com/en-in/free/), system will ask for your credit card details, but dont be alarmed as only 2 INR will get deducted and after your free trail expires, Microsoft Azure will not auto deduct any amount. Money will be ducted only if you want to extend the Azure services.

2. After creating Azure account, [create a cognitive service](https://portal.azure.com/#create/Microsoft.CognitiveServicesAllInOne), by following instruction given on the azure link.

3. Create a [Bing Image Search Resource](https://portal.azure.com/#create/Microsoft.CognitiveServicesBingSearch-v7)

4. Go to 'Keys and Endpoint' in this cognitive service. Copy one of the two keys shown there and paste it somewhere safe for a while for later use.

Now we will use this key to access Bing search engine and will download a set of 150 images for each of the class or the actress name. The code for assigining the key to environment variable is, where _XXX_ is the key you just copied  

`key = os.environ.get('AZURE_SEARCH_KEY', 'XXX')`


Lets look at a sample image that is downloaded for Jessica Chastain as a query value. First getting a list of URLs:

```
results = search_images_bing(key, 'jessica chastain')
ims = results.attrgot('content_url')
len(ims)
```

Now downloading a sample value corresponding to the first image search result -


```
dest = '/your/google/drive/location/jessica1.jpg'
download_url(ims[0], dest)
```

Displaying the results -


```
im = Image.open(dest)
im.to_thumb(128,128)
```

The sample output that we get is:
![](/images/download.png)

This seems to be working nicely till now. Next, we need to download and save 150 images each for both the actresses. The code for doing so is -


```
actress_names = 'jessica chastain', 'bryce dallas howard'
path = 'your/google/drive/location/where/you/want/save/data'
for o in actress_names:
    dest = path + o
    os.mkdir(dest)
    results = search_images_bing(key, o)
    for img in results.attrgot('content_url'):
      print(img)
      download_url(img, dest+'/'+img[-img[::-1].find('/'):])
```


Our folder has image files, as we'd expect. However, when we download images from the internet, few corrupt ones bound to get downloaded. Since this dataset is very small, you can either go and remove them manually. Or you can use the verify_images function of fastai, which I am not covering here for the brevity of the topic. Please explore on your own.


### 2. Data Loaders

Data Loaders is a class which takes the Data Loader objects passed to it, and convert them into train and validation set
