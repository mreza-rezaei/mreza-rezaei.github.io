---
permalink: /BLOG/GenerativeNetsRoadMap/
published: true
---
---
published: true
---
## Generative Adversarial Networks

### Road map for learning Generative Adeversial Networks

**what is GANs**
Generative Adversarial Networks(GAN) is a framework for training generative models that use deep neural networks. The approach simultaneously trains a generative model alongside an adversarial discriminative model. The discriminative model tries to determine whether a sample comes from the true data distribution or from the generative model, while the goal of the generative model is to fool the discriminative model.

**Generative vs Discriminative Models**

We describe the differences between discriminative and generative models. Suppose we have some data and some signal , with joint distribution . A discriminative model is a mapping from a value of to a signal . It does not care about the distribution , only that there exist some boundaries separating the 's that map to a certain value of and the 's that map to a different value of . On the other hand, a generative model tries to directly learn the distribution . In doing so, it does not explicitly learn boundaries separating different signals, but instead learns the entire distribution, which can be used to infer about the signals .

The main advantage of a generative model over discriminative models is the ability to generate samples from the distribution (supposing that the generative model is able to perfectly model the distribution). So while discriminative models are simpler to train, and typically performs better on most supervised tasks, generative models are more expressive as it approximates the true data distribution.




for theory of GANs go to [UBCCourse](http://wiki.ubc.ca/Course:CPSC522/Generative_Adversarial_Networks)


the best and most sited article in GANs is [GoodFllow](https://arxiv.org/abs/1406.2661)


in this we learn about befor works ,what is Adversial network ,theoretical results,stability of this networks ,
advantage and disadvantages ,traning results on some satabases


best blog that i find is [openAI](https://blog.openai.com/generative-models/)

**what I find on it**

there are three approaches to generative nets
	_1- [Generative Adversial Nets(GANs)](https://arxiv.org/abs/1406.2661)
    _2- [Variational Autoencoders(VAEs) ](https://arxiv.org/abs/1312.6114)
    _3- Autoregressive models


**Variational Autoencoders (VAEs)**
allow us to formalize this problem in the framework of probabilistic graphical models where we are maximizing a lower bound on the log likelihood of the data.

Variational autoencoders (VAEs) were defined in 2013 by Kingma et al. and Rezende et al



For more study see [link](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)

### Simulation VAEs with Tensorflow
see this [link](https://jmetzen.github.io/2015-11-27/vae.html)
