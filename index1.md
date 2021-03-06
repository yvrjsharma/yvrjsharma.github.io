
# 1. [Text Analysis](https://github.com/yvrjsharma/RNN)

## PART-1: IMDB Modelling Task 
![](/images/markus-winkler--fRAIQHKcc0-unsplash.jpg)

The core of this project is based around a simple task -- performing sentiment analysis with the IMDB dataset given here on [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). This dataset has 50,000 movie reviews from the IMDB corpus. 
- I have splitted these into Train (50%), Validation (30%), and Test (20%) sets.
- I’ve performed a number of different comparisons for thorough analylsis. These do not necessarily build on each other. 
- My models are designed to minimize overfitting as appropriate.
- In all cases I have recorded my results as graphs for Training and Validation data and reported the test result after training has been completed. 
- Loss functions or metrics are chosen as appropriate in a text classification task. 

#### RNN Variants
I have compared the performance on the classification tasks across **Recurrent Network Variants**. 
- I am comparing **LSTM** and **Basic RNN** models. 
- Also, I am comparing a single layer LSTM implementation to a multi-layer LSTM implementation. 

#### Embeddings 
Distributed embeddings provide a lot of power in text classification, but there are many different Embeddings types that can be used. I am comparing text classification using -
- Embeddings learned on the fly, and 
- A Pre-trained word embedding available from the Tensorflow Hub. 

#### CNN for Text Classification
**CNNs** are designed to model local features while **LSTMs** are very good at handling long range dependencies. I am investigating the use of CNNs with multiple and heterogeneous kernel sizes both as an alternative to an LSTM solution, and as an additional layer before a LSTM solution. 

#### Model Saving
From the various models created above, I am saving the model with highest accuracy. There are many ways in which models can be saved. I’m using .h5 file extensions to save a model. This file extension can save model architecture as well as mmodel weights. I have clearly documented my design in the code as well as Readme files. 

## PART-2. Working with very small Dataset (Transfer Learning use-case)
![](/images/dawit-sCKjl0AyO-4-unsplash.jpg)

A problem with libraries which provide wrappers for well-known datasets is that they can make the task of using the dataset so easy, that we do not realise what is required in the construction and use of data in Deep Learning. Whereas, in real world problems we have our own data and it is often very small. If we want to use a pre-trained models to make use of the learning that has already been achieved with an existing model, then doing this is called Transfer Learning. 

Given these issues, I have collected a very small movie review dataset and used it to train a model that is based on the pre-trained model constructed in my Part 1. 

#### Data Collection
- Firstly, I constructed a small labelled dataset of movie reviews for around randomly selected 30 movies from my birth year, that is 1987. For each movie, I selected at least one good and one bad review. 
- I encoded these reviews as positive or negative, so that the dataset can be used for processing. 

#### Modelling 
- Using the best performing model from my Part 1, I loaded the smaller dataset -- with split of 70/30 between training and validation sets. Note that, I didn't used any testing data as I don’t have enough data here in this case. 
- Then, I built a model for this novel dataset that is based on one of the previously trained model from my Part 1. This means that I fine tuned the existing model to my new data and tested its performance. In other words, this fine tuned model starts from the model that I have already saved. 
- I have reported Training and Validation scores for this fine-tuned model, and saved this model.
- Finally, I built a “from scratch” model on the smaller dataset, using the exact same architecture as my best performing model from Part 1. 
- I have compared the performance of this “from scratch” model to above fine-tuned pre-trained model.

## PART-3: Test Generation - Writing My Own Movie Reviews

![](/images/denise-jans-tV80374iytg-unsplash.jpg)

There is more to **Language Processing** then just classification. In this part I have used my skills in RNNs to **generate some original text**. I then benchmarked my Model against a more classical implementation of statistical analysis. For this work, I have again made use of the IMDB dataset of 50,000 movie reviews, except that I have split the data differently this time. My core model is based on **LSTMs**. 
Report model performance in terms of perplexity. Provide 5 outputs each from your best implementation and the statistical model. Make sure to save your best model and provide it via a link in the submission. 

- I have build one text generation model with _**only negative reviews**_, 
- One model with _**only positive reviews**_, and 
- One model with _**all reviews**_ included. 
- Lastly, with the same data, I have also implemented a **Statistical Model** for language generation.
- I have also done  comparative analysis between the four models using _**BLEU scores**_



# 2. [Deep Learning using R](https://github.com/yvrjsharma/R/blob/master/Guided_Project_DL_R_Lang.R)
## Implementing Neural Network in R using Keras
![](/images/md-mahdi-omdlGQGcoGI-unsplash.jpg)

I have implemented a small three layer Neural Network to analyse a toy image datset using Keras and R language.
The toy dataset and supporting R code can be found [here](https://github.com/yvrjsharma/R).

The toy dataset consists of tweleve images with six each of cars and aeroplanes. These are colored images and have been resized to 28 by 28 pixels for implementing a simple and fast neural network on R.
To handle image data I had to install a new package called EBImage. The process of downloading and loading can be seen in the codefile.

I have resized the images to 28x28 pixels converting the images to 28x28x3 size 3D tensors. To use Keras, we need to create tensors as inputs.
If you look at the pixel values in an image, pixels with smaller values indicate darker spots in the images. This step is called preprocesing of images.

After this, processed images are divided into training and test sets. Out of the total available twelve images, ten are kept as training while 2 are left over for training.
Five out of six car images are kept for training the model, similarly five aeroplane images are ke[t for training set.Sixth and twellveth images in dataset are kept aside as test dataset.
Train set has now dimensions as (10,2352), while test set's dimensions are (2,2352).

Now that we are ready with our datasets, we need to assign training and testing labels to these images. Since this is a case of binary classification, we will set label 1 as car, while label 0 as non-car or aeroplane.
Along with this we need to make sure that labels are also treated as tensors and not arrays or lists. For this purpose we will use to_categorical functon to convert labels into tensors as is required by Keras framework.
This process is called One Hot Encoding of labels.

After we are done with all preprocessing of images, we will start building the sequential Keras model. The activation function used throughout the neural network is Rectified Linear Unit or ReLU function.
The number of hidden units are chosen as 256 for first hidden level, 128 for next layer, and output layer has 2 units. All layers are densely connected with preceeding and following layers as well.
The output layer has Softmax function with Binary Crossentropy loss function. The optimizer I have used is ADAM and in binary classificationit is considered an industry norm in current times.
Also note that, accuracy is chosen as the metrics of calculation.

The model is trained with standard hyperparameter values of 30 epochs, 32 batch size and 20 percent validation split.
Once the model gets trained on ten training set images of cars and aeroplanes, we plot the validation and training loss against the number of epochs to verify the convergence of our model.
Since the dataset is so small, the training set prediction is hundred percent.

When the model is verified on our test data of two different class images, it gave a hundred percent accuracy rate on test data as well. This is completely due to very small train and test data.
This concludes our model building on R, and the tool that is used is RStudio. The code and toy data can be obtained from the repository given in this article.



# 3. [How you can do Image Data Augmentation using Keras](https://github.com/yvrjsharma/Image-Analysis/blob/master/Image%20Augmentation%20Using%20Keras%20ImageDataGenerator.ipynb)
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


# 4. [Creating a R Notebook for Probability and Statistical Inference](https://github.com/yvrjsharma/R/blob/master/stats.md)
![](/images/carlos-muza-hpjSkU2UYSU-unsplash.jpg)

### Understanding variables and relationships between them
In building a General Linear Statistical Model (like Linear Regression model), we first need to understand and define accurately what is happening inside the data. 
This post is written exactly to find that using statistical tools. The post finds out if there are evidences of relationships in the data and it uses appropriate statistical tests and makes decisions based on their results. 
A relationship between variables is need to have a strength value and a direction associated with it. This post identifies if there are any differential effects for different groups using statistical tests like Anova and t-test.
The analysis is based on [Student Performance dataset](https://archive.ics.uci.edu/ml/datasets/student+performance) from UCI ML repository . The R notebook explores two relationships, one between number of absences and final maths grades, other one between maths grades in period one and final maths grades. It will also check for differential effects of gender on number of absences, mother’s job on a student’s final maths grades. Lastly, it will explore if maths grades in period one and two are similar or not.

In order to find out the relationships and differences, this report conducts statistical analysis on the given dataset, and the R Notebook can be accessed by clicking on the  Title of this post or by clicking [here](https://github.com/yvrjsharma/R/blob/master/stats.md). For description of the variables present in this dataset, one can also refer this [paper](https://repositorium.sdum.uminho.pt/bitstream/1822/8024/1/student.pdf).
