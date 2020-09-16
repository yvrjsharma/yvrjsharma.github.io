---
layout: post
title: "Text Analysis using LSTM,RNN, and CNN"
date: 2020-09-15
---

Well. Finally got around to putting this old website together. Neat thing about it - powered by [Jekyll](http://jekyllrb.com) and I can use Markdown to author my posts. It actually is a lot easier than I thought it was going to be.

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
