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
So, I have build a fun web appliction using the techniques that i have learned from this chapter. My web application identifies whether the incoming image (one image at a time) is of the actress Jessica Chastain or Bryce Dallas Howard.
