## Welcome to My Portfolio


# [Text Analysis](https://github.com/yvrjsharma/RNN)

## IMDB Modelling Task 
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


2. [Deep Learning using R](https://github.com/yvrjsharma/yvrjsharma.github.io/blob/master/NeuralNetworkR.md)
 
