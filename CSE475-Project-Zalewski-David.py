#!/usr/bin/env python
# coding: utf-8

# # CSE475 Project, Due: Monday, 05/02/2022

# ## Instruction
# 
# 1. Please submit your Jupyter Notebook file (the. ipynb file) containing your code and the outputs produced by your code (note that .ipynb file can contain both the code and the outputs) to Canvas. Please name your file CSE475-Project-LastName-FirstName.ipynb.
# 
# 2. If you have any questions on the homework problems, you should post your question on the Canvas discussion board (under Project Q&A), instead of sending emails to the instructor or TA. We will answer your questions there. In this way, we can avoid repeated questions, and help the entire class stay on the same page whenever any clarification/correction is made.

# ## Handwritten Image Classifiation with Keras using MNIST and Fashion MNIST data
# 
# In this project, we will work with image data, the famous MNIST and Fashion MNIST data sets.  MNIST data set contains 70,000 images of handwritten digits (0-9) in grayscale (0=black, 255 = white). Fashion MNIST data set contains 70,000 images of clothing in grayscale (0=black, 255 = white). All the images are 28 pixels by 28 pixels for a total of 784 pixels. This is quite small by image standards. Also, the images are well centered and isolated. <br><br>
# We will use a Convolutional Neural Network and compare it with a linear neural network. 

# ### In the first part of this project, we will walk you through loading the data, building a neural network, and training it.  Then it will be your turn to work on different models.

# In[1]:


# Preliminaries

from __future__ import print_function

import keras
import tensorflow as tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from keras.datasets import fashion_mnist

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's explore the dataset a little bit

# In[2]:


# Load the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


x_train.shape


# Let's just look at a particular example to see what is inside

# In[4]:


x_train[333]  ## Just a 28 x 28 numpy array of ints from 0 to 255


# In[5]:


# What is the corresponding label in the training set?
y_train[333]


# In[6]:


# Let's see what this image actually looks like

plt.imshow(x_train[333], cmap='Greys_r')


# In[7]:


# this is the shape of the np.array x_train
# it is 3 dimensional.
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')


# In[8]:


## For our purposes, these images are just a vector of 784 inputs, so let's convert
x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)

## Keras works with floats, so we must cast the numbers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## Normalize the inputs so they are between 0 and 1
x_train /= 255
x_test /= 255


# In[9]:


# convert class labels to one-hot vectors
num_classes = 10
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

y_train[333]  # now the digit k is represented by a 1 in the kth entry (0-indexed) of the length 10 vector


# In[10]:


# We will build a model with two fully connected layers (using "Dense")
# We will use dropout of .5 to help regularize
model_1 = Sequential()
model_1.add(Dense(64, activation='relu', input_shape=(784,)))
model_1.add(Dropout(0.5))
model_1.add(Dense(64, activation='relu'))
model_1.add(Dropout(0.5))
model_1.add(Dense(10, activation='softmax'))


# In[11]:


## print the summary of the built model. Note that this model has a LOT of parameters
model_1.summary()


# In[12]:


# Let's compile the model
learning_rate = .001
model_1.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])
# note that `categorical cross entropy` is the natural generalization 
# of the loss function we had in binary classification case to multiclass case


# In[13]:


# Now let's fit (train model_1).

batch_size = 128  # mini-batch with 128 examples
epochs = 30
history = model_1.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))


# In[14]:


## We will use Keras evaluate function to evaluate performance on the test set

score = model_1.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[15]:


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["accuracy"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_accuracy"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    
plot_loss_accuracy(history)


# This is reasonably good performance, but we can do even better!  Next you will build an even bigger network and compare the performance.

# ### Keras Layers for CNNs
# - Previously we built a neural network using primarily the Dense, Activation and Dropout Layers.
# 
# - Here we will describe how to use some of the CNN-specific layers provided by Keras
# 
# #### Conv2D
# 
# ```python
# keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
# ```
# 
# A few parameters explained below for your convenience (please do check the manual of Keras to understand the meaning of all the paramters)
# - `filters`: the number of filters,  in other words, the depth of the output.
# - `kernel_size`: an (H,W) tuple giving the height and width of the kernel (filter) to be used. It can also be an integer x which is equivalent to a tuple (x,x) 
# - `strides`: and (x,y) tuple giving the stride in each dimension.  Default is `(1,1)`. It can also be an integer x which is equivalent to a tuple (x,x) 
# - `input_shape`: required only for the first layer
# 
# Note that the size of the output will be determined by the kernel_size and strides.
# 
# #### MaxPooling2D
# `keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)`
# 
# - `pool_size`: the (x,y) size of the grid to be pooled.
# - `strides`: Assumed to be the `pool_size` unless otherwise specified
# 
# #### Flatten
# Turns its input into a one-dimensional vector (per instance).  Usually used when transitioning between convolutional layers and fully connected layers.
# 

# # Your tasks for this project (three tasks, (1)-(3))
# 
# 
# ### (1) Build your own CNN model (30pts)
# Use the Keras "Sequential" functionality to build a convolutional neural network named `model_2` with the following specifications:
# <br>
# <br>
# Model Architecture:<br>
# We will build model_2 according to the famous LeNet-5 architecutre below and measure its performance.
# <br>
#     Convolution -> Relu -> Max pooling -> Convolution -> Relu -> Max pooling -> FC1 -> Relu -> FC2 -> Relu -> Output(SoftMax)
# <br>
# 
# 1. Convolution1 kernel size: 5(H) x 5(W) x 6(filters), stride = 1, no padding
# 2. Max pooling1 kernel size: 2(H) x 2(W), stride = 2
# 3. Convolution2 kernel size: 5(H) x 5(W) x 16(filters), stride = 1, no padding
# 4. Max pooling2 kernel size: 2(H) x 2(W), stride = 2
# 5. Fully Connected1 size: 120
# 6. Fully Connected2 size: 84
# 7. Train this model for 20 epochs with RMSProp at a learning rate of .001 and a batch size of 128
# 8. Plot the loss and accuracy graph for training the new model 
# 9. Evaluate the model on test data
# 
# Please train the built model on the training data (x_train, y_train), and print the test loss and test accuracy of the trained model on the test data (x_test, y_test). Please call the given plot_loss_accuracy function on the history returned by model_2.fit to have the curves of loss and accuracy on the training data and the validation data during the training process of model_2. Note that the validation data is the same as the test data.
# 
# Hints:
# <ul>
#   <li> You will need to have a Flatten layer to make the dimension of the Output be 10 (the number of classes). Check how to connect the output of convolution layer with the first fully connected layer.
# <li> You will not be graded on the accuracy of your model, but it should have a decent accuracy of at least 70%. Performance below 70% means there is something wrong with the implementation.
# <li> Do not use any padding or dropout in model_2. Such choices can be used for Project Bonus and Fashion MNIST.
#     </li></ul>

# To use the LeNet model, we need to do some preprocessing on the data first.

# In[16]:


# Data is currently flattened, we need to reshape it back to 28 * 28. To do that we reshape the data.

x_train = np.reshape(x_train, [-1, 28, 28])
x_test = np.reshape(x_test, [-1, 28, 28])
x_train.shape, x_test.shape


# In[17]:


# model_2 requires input of 32 X 32. So we will pad the train and test images with zeros to increase the size to 32 X 32.

x_train=np.pad(x_train, ((0,0), (2,2), (2, 2)), 'constant')
x_test=np.pad(x_test, ((0,0), (2,2), (2, 2)), 'constant')
x_train.shape, x_test.shape


# In[18]:


# Convolutional model requires input to be of 3 dimensions. We will add a channel dimension to it.

x_train = np.reshape(x_train, [-1, 32, 32, 1])
x_test = np.reshape(x_test, [-1, 32, 32, 1])
x_train.shape, x_test.shape


# ### Write your code below for task (1)

# In[19]:


model_2 = Sequential()
model_2.add(Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding="valid", activation="relu", input_shape=(32,32,1)))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model_2.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding="valid", activation="relu"))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model_2.add(Flatten())
model_2.add(Dense(units=120, activation='relu'))
model_2.add(Dense(units=84, activation='relu'))
model_2.add(Dense(units=10, activation = 'softmax'))
model_2.summary()

learning_rate = .001
model_2.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])

batch_size = 128  # mini-batch with 128 examples
epochs = 20
history = model_2.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))
            
plot_loss_accuracy(history)


# ### (2) Answer the following question (20pts)
# 
# Compare the curves of the loss and accuracy on the training data and the validation (test) data for model_1 and model_2. Please write down your observations. For example, (1) which model has higher validation (test) loss?  (2) which model has higher validation (test) accuracy? (3) Do (1)-(2) for the training loss and the training accuracy; (4) Which measure do you think is more meaningful, the loss or the accuracy?

# In[20]:


"""
1) model_2 had higher validation/test loss than model_1
2) model_2 had higher validation/test accuracy than model_1
3) model_1 had higher train loss than model_2
   model_2 had higher train accuracy than model_1

4) The most meaningful measure is accuracy.
"""


# ### We will perform similar tasks for Fashion MNIST dataset. 
# 
# Fashion MNIST has 10 categories of clothing items:<br>
# 
# | Label | Description | 
# | --- | --- | 
# | 0 | T-shirt/top |
# | 1 | Trouser |
# | 2 | Pullover |
# | 3 | Dress |
# | 4 | Coat |
# | 5 | Sandal |
# | 6 | Shirt |
# | 7 | Sneaker |
# | 8 | Bag |
# | 9 | Ankle boot |

# In[21]:


from __future__ import print_function

import keras
import tensorflow as tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from keras.datasets import fashion_mnist

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[22]:


x_train[0].shape


# In[23]:


#Let's just look at a particular example to see what is inside

x_train[333]  ## Just a 28 x 28 numpy array of ints from 0 to 255


# In[24]:


# What is the corresponding label in the training set?
y_train[333]


# In[25]:


# Let's see what this image actually looks like

plt.imshow(x_train[333], cmap='Greys_r')


# In[26]:


# this is the shape of the np.array x_train
# it is 3 dimensional.
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')


# In[27]:


## For our purposes, these images are just a vector of 784 inputs, so let's convert
x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)

## Keras works with floats, so we must cast the numbers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## Normalize the inputs so they are between 0 and 1
x_train /= 255
x_test /= 255


# In[28]:


# convert class vectors to binary class matrices
num_classes = 10
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

y_train[333]  


# Reperforming the earlier preprocessing methods

# In[29]:


# Data is currently flattened, we need to reshape it back to 28 * 28. To do that we reshape the data.

x_train = np.reshape(x_train, [-1, 28, 28])
x_test = np.reshape(x_test, [-1, 28, 28])
x_train.shape, x_test.shape


# In[30]:


# LeNet requires input of 32 X 32. So we will pad the train and test images with zeros to increase the size to 32 X 32.

x_train=np.pad(x_train, ((0,0), (2,2), (2, 2)), 'constant')
x_test=np.pad(x_test, ((0,0), (2,2), (2, 2)), 'constant')
x_train.shape, x_test.shape


# In[31]:


# Convolutional model requires input to be of 3 dimensions. We will add a channel dimension to it.

x_train = np.reshape(x_train, [-1, 32, 32, 1])
x_test = np.reshape(x_test, [-1, 32, 32, 1])
x_train.shape, x_test.shape


# ### (3) Training on Fashion MNIST (50pts)
# 
# Build a convolutional neural network of your choice. You can freely choose a differnet network structure, learning rate or number of epochs, which (you believe) will lead to a good model for this data set. Report the accuracy of your model on the test data (x_test,y_test).

# In[32]:


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["accuracy"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_accuracy"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)


# ### Write your code below for task (3)

# In[33]:


model_3 = Sequential()
model_3.add(Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding="valid", activation="relu", input_shape=(32,32,1)))
model_3.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model_3.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, padding="valid", activation="relu"))
model_3.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model_3.add(Flatten())
model_3.add(Dense(units=120, activation='relu'))
model_3.add(Dense(units=50, activation='relu'))
model_3.add(Dense(units=10, activation = 'softmax'))
model_3.summary()

learning_rate = .0011
model_3.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])

batch_size = 128  # mini-batch with 128 examples
epochs = 30
history = model_3.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))
            
plot_loss_accuracy(history)


# In[ ]:





# In[ ]:




