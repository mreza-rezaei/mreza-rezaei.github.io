---
#permalink: /BLOG/2017-11-18-Install-DeepLearning-Platforms-On-Windows/
published: true
---
## Install Tensorflow,Keras and Notebook on Windows

befor start any procedure you should consider 2 important things:
- if you have [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) on your laptop or pc from nvidias series, you can use it for accelerating computational tasks(But the main cost is it's difficult installation)
- windows is an suitable OS for many engineering tasks but i suggest f you want to work in  [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning)field, it is better to  use ubuntu 17.04 on your computer (  it has good support and easy installtion for required packages )

for working in Deep Leraning field you just need three things:
1.	[Keras](https://keras.io/) is a high-level neural networks API, written in Python and capable of running on top of [Tensorflow](https://www.tensorflow.org/) , [CNTK](https://github.com/Microsoft/cntk), or [Theano](https://github.com/Theano/Theano).
2.	[Tensorflow](https://www.tensorflow.org/)  is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and also used for machine learning applications such as neural networks.
3.	[Jupyter notebook](http://jupyter.org/) an environment for developing your code in python.

there are different ways for installtion, i think this is a good way to begine.

### if you have NVIDIAs GPU

1.first you should insall NVIDIAs stuff for using computational accelerator:

- CUDA® Toolkit 8.0. For details, see [NVIDIA's documentation](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) Ensure that you append the relevant Cuda pathnames to 		the `%PATH%` environment variable as described in the NVIDIA documentation.

- The NVIDIA drivers associated with CUDA Toolkit 8.0.

- cuDNN v6.1. For details, see [NVIDIA's documentation](https://developer.nvidia.com/cudnn). Note that cuDNN is typically installed in a different location from the other CUDA DLLs. Ensure that you add the directory where you installed the cuDNN DLL to your `%PATH%` environment variable.

- GPU card with CUDA Compute Capability 3.0 or higher. See NVIDIA documentation for a list of supported GPU cards.

2.you should install [Anaconda](https://www.anaconda.com/download/) for easy management and work with packages.

3.in this step you just need install desired packages:

after install Anaconda a terminal became open, now you should install this packages:
- Tensorflow package :
use command ``` conda install -c anaconda tensorflow-gpu``` in terminal
- Keras package :
use command ``` conda install -c anaconda keras-gpu``` in terminal
- notebooke package :
use command ``` conda install -c anaconda notebook ``` in terminal


### if you dont have NVIDIAs GPU

1.you should install [Anaconda](https://www.anaconda.com/download/) for easy management and work with packages.

2.in this step, you just need installing desired packages:

after installation of Anaconda, a terminal will be opened. After that, you shoud install this packages:
- Tensorflow package :
use command ``` conda install -c anaconda tensorflow``` in terminal
- Keras package :
use command ``` conda install -c anaconda keras``` in terminal
- notebooke package :
use command ``` conda install -c anaconda notebook ``` in terminal



### getting started with jupyter notebook

after installtion is completed, you can use jupyter notebook env for your coding.
use ```jupyter notebook``` command in terminal.
for starting with jupyter notebook you can use [this](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb) tutorial

### test correct installtion of Tensorflow & keras

1.open jupyter notebook and wrie following code in a cell and run it
``` import tensorflow as tf ```
if the installtion is correct you shouldn't see any error.

2.wrie following code in a cell and run it
``` import keras ```
if the installtion is correct you shouldn't see any error.
