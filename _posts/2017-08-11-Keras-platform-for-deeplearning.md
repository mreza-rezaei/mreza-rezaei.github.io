---
published: true
---
## Keras platform for deep learning

in this post i'm goinig to show you  :

- what is keras and how creat a neural network with that

- the wide range ability of keras platform 

- create CNN  models with keras

- use pretrained models and weights 

- fine-tuning the pretrained networks 

### what is keras and how creat a neural network with that

[Keras](https://keras.io/) is a high-level neural networks API, written in Python and capable of running on top of [TensorFlow](), [CNTK](), or [Theano](). It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

for more reading and installtion jump to [this](https://keras.io/)

#### create a neural network

in this part the code provided for create a MLP model for pima Indians onset of diabetes dataset,
that's can download from [here](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)


 ```python
 
# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
model.add(Dense(8, init= uniform , activation= relu ))
model.add(Dense(1, init= uniform , activation= sigmoid ))
# Compile model
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

```

###  the wide range ability of keras platform  

there are a lot of decisions to make when designing and configuring your deep learning models,keras provide large set of this decisions for us to use them simply
 - Data splitting
 
Keras provides two convenient ways of evaluating your deep learning
algorithms this way:
1. Use an automatic verification dataset.

Keras can separate a portion of your training data into a validation dataset and evaluate the
performance of your model on that validation dataset each epoch.

 ```python
 
# MLP with automatic validation set
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
model.add(Dense(8, init= uniform , activation= relu ))
model.add(Dense(1, init= uniform , activation= sigmoid ))
# Compile model
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10) 

```



2. Use a manual verification dataset.
Keras also allows you to manually specify the dataset to use for validation during training.
In this example we use the handy train test split() function from the Python scikit-learn
machine learning library to separate our data into a training and test dataset



 ```python
 
# MLP with manual validation set
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
model.add(Dense(8, init= uniform , activation= relu ))
model.add(Dense(1, init= uniform , activation= sigmoid ))
# Compile model
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=150, batch_size=10)

```



 - Manual k-Fold Cross Validation
 
The gold standard for machine learning model evaluation is k-fold cross validation. It provides
a robust estimate of the performance of a model on unseen data. It does this by splitting the
training dataset into k subsets and takes turns training models on all subsets except one which
is held out, and evaluating model performance on the held out validation dataset. The process
is repeated until all subsets are given an opportunity to be the held out validation set. The
performance measure is then averaged across all models that are created.

 ```python
 
 MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
model.add(Dense(8, init= uniform , activation= relu ))
model.add(Dense(1, init= uniform , activation= sigmoid ))
# Compile model
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
# Fit the model
model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X[test], Y[test], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

 ```

 - Save Your Models For Later With
Serialization

first install h5py with pip

`` $sudo pip install h5py ``

JSON is a simple file format for describing data hierarchically. Keras provides the ability to
describe any model using JSON format with a to json() function. This can be saved to file
and later loaded via the model from json() function that will create a new model from the
JSON specification.
The weights are saved directly from the model using the save weights() function and
later loaded using the symmetrical load weights() function. The example below trains and
evaluates a simple model on the Pima Indians dataset. The model structure is then converted
to JSON format and written to model.json in the local directory. The network weights are
written to model.h5 in the local directory.
The model and weight data is loaded from the saved files and a new model is created. It is
important to compile the loaded model before it is used. This is so that predictions made using
the model can use the appropriate ecient
computation from the Keras backend. The model is
evaluated in the same way printing the same evaluation score.
assification accuracy on the validation dataset (monitor=’val acc’ and mode=’max’). The
weights are stored in a file that includes the score in the filename
weights-improvement-val acc=.2f.hdf5

 ```python
 
# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
model.add(Dense(8, init= uniform , activation= relu ))
model.add(Dense(1, init= uniform , activation= sigmoid ))
# Compile model
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# later...
# load json and create model
json_file = open( model.json , r )
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss= binary_crossentropy , optimizer= rmsprop , metrics=[ accuracy ])
score = loaded_model.evaluate(X, Y, verbose=0)
print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)

```

you can also use YAML for saving model

 - Keep The Best Models During
 
Training With Checkpointing
Application checkpointing is a fault tolerance technique for long running processes. It is an
approach where a snapshot of the state of the system is taken in case of system failure.
A good use of checkpointing is to output the model weights each time an improvement is
observed during training.

 ```python 
 
# Checkpoint the weights when validation accuracy improves
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
model.add(Dense(8, init= uniform , activation= relu ))
model.add(Dense(1, init= uniform , activation= sigmoid ))
# Compile model
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor= val_acc , verbose=1, save_best_only=True,
mode= max )
callbacks_list = [checkpoint]
# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,
callbacks=callbacks_list, verbose=0)

```


A simpler checkpoint strategy is to save the model weights to the same file, if and only if the
validation accuracy improves. This can be done easily using the same code from above and
changing the output filename to be fixed (not include score or epoch information). In this case,
model weights are written to the file weights.best.hdf5 only if the classification accuracy of
the model on the validation dataset improves over the best seen so far.


 ```python
 
 # Checkpoint the weights for best model on validation accuracy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
model.add(Dense(8, init= uniform , activation= relu ))
model.add(Dense(1, init= uniform , activation= sigmoid ))
# Compile model
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor= val_acc , verbose=1, save_best_only=True,
mode= max )
callbacks_list = [checkpoint]
# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,
callbacks=callbacks_list, verbose=0)

```


 -  Loading a Saved Neural Network Model
 Now that you have seen how to checkpoint your deep learning models during training, you need
to review how to load and use a checkpointed model. The checkpoint only includes the model
weights. It assumes you know the network structure. This too can be serialize to file in JSON
or YAML format. In the example below, the model structure is known and the best weights are
loaded from the previous experiment, stored in the working directory in the weights.best.hdf5
file. The model is then used to make predictions on the entire dataset.


 ```python
 
# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
model.add(Dense(8, init= uniform , activation= relu ))
model.add(Dense(1, init= uniform , activation= sigmoid ))
# load weights
model.load_weights("weights.best.hdf5")
# Compile model (required to make predictions)
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
print("Created model and loaded weights from file")
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

```

 - Understand Model Behavior During
Training By Plotting History
We can create plots from the collected history data. In the example below we create a
small network to model the Pima Indians onset of diabetes binary classification problem . The example collects the history, returned from training the model and creates
two charts:
1. A plot of accuracy on the training and validation datasets over training epochs.
2. A plot of loss on the training and validation datasets over training epochs.

 ```python
 
# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init= uniform , activation= relu ))
model.add(Dense(8, init= uniform , activation= relu ))
model.add(Dense(1, init= uniform , activation= sigmoid ))
# Compile model
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history[ acc ])
plt.plot(history.history[ val_acc ])
plt.title( model accuracy )
plt.ylabel( accuracy )
plt.xlabel( epoch )
15.2. Visualize Model Training History in Keras 100
plt.legend([ train , test ], loc= upper left )
plt.show()
# summarize history for loss
plt.plot(history.history[ loss ])
plt.plot(history.history[ val_loss ])
plt.title( model loss )
plt.ylabel( loss )
plt.xlabel( epoch )
plt.legend([ train , test ], loc= upper left )
plt.show()

```

 - Reduce Overfitting With Dropout
Regularization
Dropout is a regularization technique for neural network models proposed by Srivastava, et al.
in their 2014 paper Dropout: A Simple Way to Prevent Neural Networks from Overfitting1.
Dropout is a technique where randomly selected neurons are ignored during training. They
are dropped-out randomly. This means that their contribution to the activation of downstream
neurons is temporally removed on the forward pass and any weight updates are not applied to
the neuron on the backward pass.

 ```python 
 
# Example of Dropout on the Sonar Dataset: Hidden Layer
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# dropout in hidden layers with weight constraint
def create_model():
# create model
model = Sequential()
model.add(Dense(60, input_dim=60, init= normal , activation= relu ,
W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(30, init= normal , activation= relu , W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(1, init= normal , activation= sigmoid ))
# Compile model
sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss= binary_crossentropy , optimizer=sgd, metrics=[ accuracy ])
return model
numpy.random.seed(seed)
estimators = []
estimators.append(( standardize , StandardScaler()))
estimators.append(( mlp , KerasClassifier(build_fn=create_model, nb_epoch=300,
batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

```


 - Lift Performance With Learning Rate
 
Schedules
Adapting the learning rate for your stochastic gradient descent optimization procedure can
increase performance and reduce training time. Sometimes this is called learning rate annealing
or adaptive learning rates. Here we will call this approach a learning rate schedule, were the
default schedule is to use a constant learning rate to update network weights for each training
epoch.
two approch:
1-Time-Based Learning Rate Decay
2-Drop-Based Learning Rate Decay

 ```python
 
import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("ionosphere.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
# create model
model = Sequential()
model.add(Dense(34, input_dim=34, init= normal , activation= relu ))
model.add(Dense(1, init= normal , activation= sigmoid ))
# Compile model
epochs = 50
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss= binary_crossentropy , optimizer=sgd, metrics=[ accuracy ])
# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=epochs, batch_size=28, verbose=2)

```

### create CNN  models with keras
in pefor [post](https://mreza-rezaei.github.io/CNN-For-Classification-Digits-On-MNIST/) i create a model for Handwritten Digits on MNIST dataset and use tensorflow ,in this part i'm escaple from expalin wha is CNN and jump to create below model of CNN in keras 
convlutiional network model layers:

1. Convolutional layer with 30 feature maps of size 5 ⇥ 5.
2. Pooling layer taking the max over 2 ⇥ 2 patches.
3. Convolutional layer with 15 feature maps of size 3 ⇥ 3.
4. Pooling layer taking the max over 2 ⇥ 2 patches.
5. Dropout layer with a probability of 20%.
6. Flatten layer.
7. Fully connected layer with 128 neurons and rectifier activation.
19.5. Larger Convolutional Neural Network for MNIST 132
8. Fully connected layer with 50 neurons and rectifier activation.
9. Output layer


 ```python
 
# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering( th )
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype( float32 )
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype( float32 )
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define the larger model
def larger_model():
# create model
model = Sequential()
model.add(Convolution2D(30, 5, 5, input_shape=(1, 28, 28), activation= relu ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(15, 3, 3, activation= relu ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation= relu ))
model.add(Dense(50, activation= relu ))
model.add(Dense(num_classes, activation= softmax ))
# Compile model
model.compile(loss= categorical_crossentropy , optimizer= adam , metrics=[ accuracy ])
return model
# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200,
verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

```


 - Improve Model Performance With
Image Augmentation
we can improve our model with Augmentation datas we can do that with below methods
1-Feature Standardization

 ```python
 
# Standardize images across the dataset, mean=0, stdev=1
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load data
20.3. Feature Standardization 138
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype( float32 )
X_test = X_test.astype( float32 )
# define data preparation
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
# create a grid of 3x3 images
for i in range(0, 9):
pyplot.subplot(330 + 1 + i)
pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap( gray ))
# show the plot
pyplot.show()

break```

2-ZCA Whitening
A whitening transform of an image is a linear algebra operation that reduces the redundancy
in the matrix of pixel images. Less redundancy in the image is intended to better highlight
the structures and features in the image to the learning algorithm

 ```python
 
# ZCA whitening
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype( float32 )
X_test = X_test.astype( float32 )
# define data preparation
datagen = ImageDataGenerator(zca_whitening=True)
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
# create a grid of 3x3 images
for i in range(0, 9):
pyplot.subplot(330 + 1 + i)
pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap( gray ))
# show the plot
pyplot.show()
break

```

3- Random Rotations

 ```python
 
# Random Rotations
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype( float32 )
X_test = X_test.astype( float32 )
# define data preparation
datagen = ImageDataGenerator(rotation_range=90)
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
# create a grid of 3x3 images
for i in range(0, 9):
pyplot.subplot(330 + 1 + i)
pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap( gray ))
# show the plot
pyplot.show()
break

```

4- Random Shifts

 ```python
 
 Random Shifts
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype( float32 )
X_test = X_test.astype( float32 )
# define data preparation
shift = 0.2
20.6. Random Shifts 143
datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
# create a grid of 3x3 images
for i in range(0, 9):
pyplot.subplot(330 + 1 + i)
pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap( gray ))
# show the plot
pyplot.show()
break

```

5-Random Flips

 ```python
 
# Random Flips
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype( float32 )
X_test = X_test.astype( float32 )
# define data preparation
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
# create a grid of 3x3 images
for i in range(0, 9):
pyplot.subplot(330 + 1 + i)
pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap( gray ))
# show the plot
pyplot.show()
break

```

after doing augmentation on dataset yo can save images to the file

 ```python
 
# Save augmented images to file
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
from keras import backend as K
K.set_image_dim_ordering( th )
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype( float32 )
X_test = X_test.astype( float32 )
# define data preparation
datagen = ImageDataGenerator()
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
os.makedirs( images )
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir= images ,
save_prefix= aug , save_format= png ):
# create a grid of 3x3 images
for i in range(0, 9):
pyplot.subplot(330 + 1 + i)
pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap( gray ))
# show the plot
pyplot.show()
break
 
 ```

### use pretrained models and weights 
in this [git](https://github.com/mreza-rezaei/deep-learning-models) you can access to the some famous models that trained on image net 
you can download the models and pretrained weigths from there

how we can use of pretrained networks:
first download the desierd network model in keras (i choose VGG16 for my work)

then use below code to 

 ```python
 
# USAGE
# python test_imagenet.py --image images/dog_beagle.png

# import the necessary packages
from matplotlib import pyplot as plt
from keras.preprocessing import image as image_utils
import VGG16
from keras.utils.data_utils import get_file
import numpy as np
import json
import argparse

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def decode_predictions(preds):
    global CLASS_INDEX
    assert len(preds.shape) == 2 and preds.shape[1] == 1000
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    indices = np.argmax(preds, axis=-1)
    results = []
    for i in indices:
        results.append(CLASS_INDEX[str(i)])
    return results
    
    # load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pxiels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
print("[INFO] loading and preprocessing image...")
image = image_utils.load_img('patch to image', target_size=(224, 224))
image = image_utils.img_to_array(image)
print(image.shape)
plt.imshow(image)
# our image is now represented by a NumPy array of shape (3, 224, 224),
# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
# pass it through the network -- we'll also preprocess the image by
# subtracting the mean RGB pixel intensity from the ImageNet dataset
image = np.expand_dims(image, axis=0)
# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet")

# classify the image
print("[INFO] classifying image...")
preds = model.predict(image)
(inID, label) = decode_predictions(preds)[0]
# display the predictions to our screen
print("ImageNet ID: {}, Label: {}".format(inID, label))

```

### fine-tuning the pretrained networks
for fine tuning you just need to follow this steps:
1- load model
2- load the weights
3- cut the fully connected layer
4- add new fully connected layer
5- freez the earllier layers(alternative)
6- train new model

in the below example we use pretrained model of vgg16 that trained on Imagenet and use it to classificate cifar-10

```python

 from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss

from load_cifar10 import load_cifar10_data

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Model for Keras
    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('imagenet_models/vgg16_weights.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    #  set the first 10 layers to non-trainable (weights will not be updated)
    for layer in model.layers[:10]:
        layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 10 
    batch_size = 16 
    nb_epoch = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    # Load our model
    model = vgg16_model(img_rows, img_cols, channel, num_classes)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
score = log_loss(Y_valid, predictions_valid)

