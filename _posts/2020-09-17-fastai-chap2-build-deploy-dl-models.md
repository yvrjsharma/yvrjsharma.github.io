---
layout: post
title: "Gathering Data, Building a Computer Vision model, and Deploying the model"
date: 2020-09-17
---
![](/images/photos-hobby-g29arbbvPjo-unsplash.jpg)

## Why this article

Hi Everyone, so this is my first blog on the [Fastai](https://www.amazon.in/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) book that I have started reading recently. It has been a fascinating read so far. I am have just completed my second chapter, and I have already built a high quality deep learning image classificatoion model. Though, I am not a beginner in this field, and I understand the math and programming architecture very well behind the state of the art networks used in  building the model (Resne18 in this particular case), this chapter basically teaches you how to build a DL model with very few lines of codes and also deploy it as a web application. This last part of deploying the model has what fascinated me the most in this chapter. So, I figured that most of you might not be interested in putting hours behind this book, due to your already busy schedules, it will be beneficial for you to read this blog and use this to build your first model along with deploying it on the web. Jeremy and Sylvain's book has a dedicated website too on which you can simply go and read all the cahpters and copy code to run in your own environment. Still, oit leaves out a lot of work for you to figure out as you go. For example, I encountered a number of errors and confusions while impleneting this particular chapter and would be glad to share my learning through this blog, so that anyone who is not reading the book, or who has read it and is stuck somewhere, can use this articlee to sort them out.

## Building a Computer Vision model

              ![](/images/chap2_fastai.jpeg)

So, I have build a fun web appliction using the techniques that i have learned from this chapter. My web application identifies whether the incoming image (one image at a time) is of the actress Jessica Chastain or Bryce Dallas Howard. So, the idea is to identify an incoming image to onee of the two computer vision classes. Now, I could have gone with classiffying between images of Cats and Dogs or between different food items or between different clothing items, but I chose this particular case of doppelganger as these two actress have confused me previously. I used to get confused betwenn the two them and this has been acknowledged by various online publications as well. Okay, fun apart, this has been an interesting application to develop and even though my model was abel to predict with 91% validation accuracy between the two actresses, new incoming images of Bryce Dallas Howard were often confused with Jessica Chastain, confirming my doubts that sometimes it is [very difficult to identify between them who's who](https://www.today.com/video/can-you-tell-jessica-chastain-and-bryce-dallas-howard-apart-461250627836).

## Deploying the model
![](/images/mybinder.png)

I have used Binder, which is a **free** and _simplest_ method out there to publish your web app online. This allows you to host a POC type or a hobby type project of yours for bigger community to see easily by clicking on a link. I will be sharing below my web application's link as well as how you can do the same. Please keep in mind that this is a free support available online, so it might suffer some glitches or lag time when you will try to access my application online.
