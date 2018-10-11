---
published: true
---
# Generative Models


## Learning Goals

	- [Introduction]
	- [Generative Adversarial Networks]
    - [Variational Autoencoders]
    - [DCGAN]
    - [Extensions]


# introduction
in this tutorial we going to learn about Generative Models and implement some of them

what is Generative Model:

In probability and statistics, a generative model is a model for randomly generating observable data values, typically given some hidden parameters. It specifies a joint probability distribution over observation and label sequences. Generative models are used in machine learning for either modeling data directly (i.e.https://arxiv.org/abs/1312.6114, modeling observations drawn from a probability density function), or as an intermediate step to forming a conditional probability density function. A conditional distribution can be formed from a generative model through Bayes' rule.


# generative Adversial Network

Generative Adversarial Networks ([GAN](https://arxiv.org/abs/1406.2661)) is a framework for training generative models that use deep neural networks. The approach simultaneously trains a generative model alongside an adversarial discriminative model. The discriminative model tries to determine whether a sample comes from the true data distribution or from the generative model, while the goal of the generative model is to fool the discriminative model.

![GAN](http://www.kdnuggets.com/wp-content/uploads/generative-adversarial-network.png)

 ## Generative vs Discriminative Models

We describe the differences between discriminative and generative models. Suppose we have some data and some signal , with joint distribution . A discriminative model is a mapping from a value of to a signal . It does not care about the distribution , only that there exist some boundaries separating the 's that map to a certain value of and the 's that map to a different value of . On the other hand, a generative model tries to directly learn the distribution . In doing so, it does not explicitly learn boundaries separating different signals, but instead learns the entire distribution, which can be used to infer about the signals .

The main advantage of a generative model over discriminative models is the ability to generate samples from the distribution (supposing that the generative model is able to perfectly model the distribution). So while discriminative models are simpler to train, and typically performs better on most supervised tasks, generative models are more expressive as it approximates the true data distribution.

Below, we discuss a framework that uses neural networks to construct a generative model. Neural networks have been shown to perform spectacularly as discriminative models, usually in a classification setting where the inputs are high dimensional. GAN is a method that takes advantage of the performance of neural networks as discriminative models to aid in the training of a generative neural network.



# Deep-Conv-GAN

One such recent model is the DCGAN network from [Radford](https://github.com/Newmu/dcgan_code) et al. (shown below). This network takes as input 100 random numbers drawn from a uniform distribution (we refer to these as a code, or latent variables, in red) and outputs an image (in this case 64x64x3 images on the right, in green). As the code is changed incrementally, the generated images do too — this shows the model has learned features to describe how the world looks, rather than just memorizing some examples.

The network (in yellow) is made up of standard convolutional neural network components, such as deconvolutional layers (reverse of convolutional layers), fully connected layers, etc.:
![DCGAN](https://blog.openai.com/content/images/2017/02/gen_models_diag_1.svg)

DCGAN is initialized with random weights, so a random code plugged into the network would generate a completely random image. However, as you might imagine, the network hasmillions of parameters that we can tweak, and the goal is to find a setting of these parameters that makes samples generated from random codes look like the training data. Or to put it another way, we want the model distribution to match the true data distribution in the space of images.


# Variational-Autoencoders

Variational Autoencoders ([VAEs](https://arxiv.org/abs/1312.6114)) allow us to formalize this problem in the framework of probabilistic graphical models where we are maximizing a lower bound on the log likelihood of the data.

Variational autoencoders (VAEs) were defined in 2013 by Kingma et al. and Rezende et al
 
 Some simulation result with tensorflow
 ![vae1](https://mreza-rezaei.github.io/images/Vae1.png)
 ![vae2](https://mreza-rezaei.github.io/images/Vae2.png)
 
 
 
 # applications
 
 1-[Adversarial-training-and-dilated-convolutions-for-brain-MRI-segmentation](https://arxiv.org/abs/1707.03195)


