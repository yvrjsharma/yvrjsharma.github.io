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

Data Loaders is a class which takes the Data Loader objects passed to it, and convert them into train and validation set.

```
actress = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

```
To turn our downloaded into dataloaders object, fastai needs atleast four different things -
1. What kind of data we are working with. This tuple specify what type of data we want as dependent and independent variables respectively.  
`blocks=(ImageBlock, CategoryBlock)`
2. How to get the list of items. `get_image_files` takes a path and returns a list of all of the images in that path.
3. How to label these items. `get_y=parent_label` givesthe name of the folder in which images are kept as the class label.
4. How to create the validation set. `RandomSplitter(valid_pct=0.2, seed=42)` gives you train and validation sets by splitting the availabel data into 80-20% respectively.

We feed images in our model in batches to train on them. As such, the size of eac image should be same in a batch. This is taken care by item_transform functions, and here we are passing it size as 128 by 128 pixels.

Actual source of data is given to the dataloaders object as shown below -

`dls = actress.dataloaders(path)`

We can also look at a sample of images from validation or training dataloader objects as shown (code and output) -

`dls.valid.show_batch(max_n=4, nrows=1)`

![](/image/ou1.png)

Since our dataset is small, we need to augment the size by doing operations on images which create a new image without changing the meanining or affetcting the class labels. Examples of such operationsare - flipping an image horrizontaly and vertically, brightness and contrast changes, shearing an image or warping it at an angle.

The code that I am using for this augmentation is -

```
actress = actress.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = actress.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```

Note that, here I am applying batch transformation by using `batch_tfms` parameter. `aug_transforms` works best for the set of augmentation operations listed above. I am fastai enables the parameter tuning very easy as I am simply using the dataloader object and decribing or adding new parameters on it. The output that we get is -

![](/images/out2.png)

### 3. Training the  model

I will now create the `learner` and then fine-tune it -

```
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)
```

Lets understand these code lines. First line instructs fastai to create a convolutional neural network (CNN) and specifies the architecture fastai should use (`resnet18`), this will be the kind of model that fastai will create. `Resnet` is a standard architecture which is both fast and accurate for most of the image datasets and problems. The 18 in `resnet18` refers to the depth or the number of layers in the network. The other options are 50,101 and 152. `metrics=error_rate` is meant for measuring the quality of the model using the validation set. This metric value will be printed at the end of each epoch. An epoch is one complete round of the model through the image dataset.

There is one more parameter called `pretrained` which is default set to `true`. This would mean that `resnet18` is already trained on `ImageNet` dataset. ImageNet refers to the large image dataset consisting of more than 1.2 million images divided into 1000 classes. Each year a classification competition is held which is called ILSVRC, in which individuals and teams take part to build classification and detection models. Historically the `error rate` has decreased in every competition year. So, this `resnet18` is trained on this large dataset and had attained very high state-of-the-art performance on it. Thus the wweights or parameters of this `resnet` model are pretrained or learned from the previous task only. Using a pretrained model into a new classification task is called **Tansfer Learning**. Using such pretrained models, reduces the training time as well as improve the model performance manifolds.

Second line `learn.fine_tune(4)` implies that a _pretrained model_ is used on a new dataset, or it  will be _fine tuned_ on the new dataset. Pretrained or existing weights are not disturbed and only last few layers are trained on the new data. I will write a separate post later explaining the reasoning behind and performance of transfer learning using fine tuning.

Lets look at the performance of our model, by plotting the confusion metrics -

```
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

![](/images/out3.png)

Now, fastai comes with wonderful functionality of identifying the images which have high loss values as well as which are incorrectly classified by our model. As we can see there are a total of 5 incorrect classifications. Using below code line, I am displaying the  images with high loss values, along with their predicted label, followed by actual label, loss value and lastly predicting accuracy.

![](/images/out4.png)  

Sometimes, and since our dataset size is very small, we can make good use of this information and can counter check if the validation set has incorrect actual labels or not.
The book suggests that one should follow this intuitive approach to doing data cleaning. Build a quick, simple and small model first, and then use it to help identify incorrectly identified actual class labels as suggested above. There is also a handy GUI data cleaner that comes with fastai bundle and it is called `ImageClassifierCleaner`, but more on it maybe later in another post.  

### Turning trained Model into an Online Application

**Saving Model**

A model consists of two parts - architecture and the trained weights. Easy way to save a model is to save both of them together, that way you don't have to load them separately at the time of predicting results on new value or drwaing an inference. To save both elements together in one go, fastai uses `export` method. Using method fastai saves a pickle file or a serialized file.

```
save_path = Path('/path/gdrive/to/your/location/')
learn.export(save_path/'actress.pkl')
```

**Using Model as Inference**

We can use `load_learner` method of fastai to reload a model back into memory and then use it draw _inference_ on an incoming image. To do this, just pass the filename and path to `predict` method of fastai. It will return three items - predicted label class, index of predicted label, and probabilities of each class.

`learn_inf.predict(save_path/'jessica-sample.jpg')`  

Output will be -

`('jessica', tensor(1), tensor([0.0211, 0.9789]))`

**Building an App**

For this part, you don't have to be proficient in web development. Know that, you can create a complete working app using only your jupyter notebook. Exciting, isn't it? What makes this possible is Ipython widgets called Ipywidgets and a renderer called Voila. Voila allows users to avoid running jupyter notebook from their front-end. It converts the entire notebok into a deployable web-application. While, Ipywidgets helps in building buttons as a part of our raphical user interface.

```
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()
btn_run = widgets.Button(description='Classify')


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)

```

First line of this code, uses widgets to create a button to upload images, and second displays the image on the screen. Label widget helps in displaying the assigned labels from model inference, and create another buton which will classify the image or will run the inference on our image. Defined function acts as a _click event handler_ on classify button. Putting entire thing into a single box to display all things together -

```
#hide_output
VBox([widgets.Label('Select the Actress!'),
      btn_upload, btn_run, out_pl, lbl_pred])

```
output is -

![](/images/out5.png)

Now, please go ahead and install Voila and enable the jupyter extension as shown -

```
!pip install voila
!jupyter serverextension enable voila —sys-prefix

```

**Deploying your Model**
