
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


# [Deep Learning using R](https://github.com/yvrjsharma/yvrjsharma.github.io/blob/master/NeuralNetworkR.md)
## Implementing Neural Network in R using Keras
![](/images/md-mahdi-omdlGQGcoGI-unsplash.jpg)

I have implemented a small three layer Neural Network to analyse a toy image datset using Keras and R language.
The toy dataset and supporting R code can be found [here](https://github.com/yvrjsharma/R).

The toy dataset consists of tweleve images with six each of cars and aeroplanes. These are colored images and have been resized to 28 by 28 pixels for implementing a simple and fast neural network on R.
To handle image data I had to install a new package called EBImage. The process of downloading and loading can be seen in the codefile.

I have resized the images to 28*28 pixels converting the images to 28*28*3 size 3D tensors. To use Keras, we need to create tensors as inputs.
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

