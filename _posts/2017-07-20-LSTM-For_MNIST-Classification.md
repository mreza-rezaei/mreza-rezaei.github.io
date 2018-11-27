---

published: true
---

# A Recurrent Neural Network (LSTM) For Classification MNIST Dataset in Tensorflow

# Recurrent Networks

Up until now, all of the networks that we've learned and worked with really have no sense of time.  They are static.  They cannot remember sequences, nor can they understand order outside of the spatial dimensions we offer it.  Imagine for instance that we wanted a network capable of reading.  As input, it is given one letter at a time.  So let's say it were given the letters 'n', 'e', 't', 'w', 'o', 'r', and we wanted it to learn to output 'k'.   It would need to be able to reason about inputs it received before the last one it received, the letters before 'r'.  But it's not just letters.

Consider the way we look at the world.  We don't simply download a high resolution image of the world in front of us.  We move our eyes.  Each fixation takes in new information and each of these together in sequence help us perceive and act.  That again is a sequential process.

Recurrent neural networks let us reason about information over multiple timesteps.  They are able to encode what it has seen in the past as if it has a memory of its own.  It does this by basically creating one HUGE network that expands over time.  It can reason about the current timestep by conditioning on what it has already seen.  By giving it many sequences as batches, it can learn a distribution over sequences which can model the current timestep given the previous timesteps.  But in order for this to be practical, we specify at each timestep, or each time it views an input, that the weights in each new timestep cannot change.  We also include a new matrix, `H`, which reasons about the past timestep, connecting each new timestep.  For this reason, we can just think of recurrent networks as ones with loops in it.

Other than that, they are exactly like every other network we've come across!  They will have an input and an output.  They'll need a loss or an objective function to optimize which will relate what we want the network to output for some given set of inputs.  And they'll be trained with gradient descent and backprop.

<a name="basic-rnn-cell"></a>
## Basic RNN Cell

The basic recurrent cell can be used in tensorflow as `tf.contrib.rnn.BasicRNNCell`.  Though for most complex sequences, especially longer sequences, this is almost never a good idea.  That is because the basic RNN cell does not do very well as time goes on.  To understand why this is, we'll have to learn a bit more about how backprop works.  When we perform backrprop, we're multiplying gradients from the output back to the input.  As the network gets deeper, there are more multiplications along the way from the output to the input.

Same for recurrent networks.  Remember, their just like a normal feedforward network with each new timestep creating a new layer.  So if we're creating an infinitely deep network, what will happen to all our multiplications?  Well if the derivatives are all greater than 1, then they will very quickly grow to infinity.  And if they are less than 1, then they will very quickly grow to 0. That makes them very difficult to train in practice.  The problem is known in the literature as the exploding or vanishing gradient problem.  Luckily, we don't have to figure out how to solve it, because some very clever people have already come up with a solution, in 1997!, yea, what were you doing in 1997.  Probably not coming up with they called the long-short-term-memory, or LSTM.

## LSTM RNN Cell

The mechanics of LSTM,  it uses a combinations of gating cells to control its contents and by having gates, it is able to block the flow of the gradient, avoiding too many multiplications during backprop.  For more details, I highly recommend reading: https://colah.github.io/posts/2015-08-Understanding-LSTMs/.

## import dataset


```python
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST a
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

```


```python
print(mnist.train.images.shape)
print(mnist.validation.images.shape)
print(mnist.test.images.shape)
```

- Training (mnist.train) >>  Use the given dataset with inputs and related outputs for training of NN. In our case, if you give an image that you know that represents a "nine", this set will tell the neural network that we expect a "nine" as the output.  
        - 55,000 data points
        - mnist.train.images for inputs
        - mnist.train.labels for outputs
  
   
- Validation (mnist.validation) >> The same as training, but now the date is used to generate model properties (classification error, for example) and from this, tune parameters like the optimal number of hidden units or determine a stopping point for the back-propagation algorithm  
        - 5,000 data points
        - mnist.validation.images for inputs
        - mnist.validation.labels for outputs
  
  
- Test (mnist.test) >> the model does not have access to this informations prior to the test phase. It is used to evaluate the performance and accuracy of the model against "real life situations". No further optimization beyond this point.  
        - 10,000 data points
        - mnist.test.images for inputs
        - mnist.test.labels for outputs

To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.


```python
# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

```

### Creating placeholders and Set parameters

It's a best practice to create placeholders before variable assignments when using TensorFlow. Here we'll create placeholders for inputs ("Xs") and outputs ("Ys").   

__Placeholder 'X':__ represents the "space" allocated input or the images. 
       
       * 1st dimension = None. Indicates that the batch size, can be of any size.  
       * 2nd dimension = timesteps 
       * 3nd dimension =MNIST data input (img shape: 28*28)
      
__Placeholder 'Y':___ represents the final output or the labels.  
       * 10 possible classes (0,1,2,3,4,5,6,7,8,9)  
       * The 'shape' argument defines the tensor size by its dimensions.  
       * 1st dimension = None. Indicates that the batch size, can be of any size.   
       * 2nd dimension = 10. Indicates the number of targets/outcomes 

__dtype for both placeholders:__ if you not sure, use tf.float32. The limitation here is that the later presented softmax function only accepts float32 or float64 dtypes. For more dtypes, check TensorFlow's documentation <a href="https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#tensor-types">here</a>



```python
# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

```

The dimensions for data are [Batch Size, Sequence Length, Input Dimension]. We let the batch size be unknown and to be determined at runtime. Target will hold the training output data which are the correct results that we desire. We’ve made Tensorflow placeholders which are basically just what they are, placeholders that will be supplied with data later.

#### Weights and Biases


```python
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

```

## define the network


```python
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

```

<a id="ref8"></a>
# Define functions and train the model

#### Define the loss function

We need to compare our output, layer4 tensor, with ground truth for all mini_batch. we can use __cross entropy__ to see how bad our CNN is working - to measure the error at a softmax layer.

The following code shows an toy sample of cross-entropy for a mini-batch of size 2 which its items have been classified. You can run it (first change the cell type to __code__ in the toolbar) to see hoe cross entropy changes.

#### Define the optimizer

It is obvious that we want minimize the error of our network which is calculated by cross_entropy metric. To solve the problem, we have to compute gradients for the loss (which is minimizing the cross-entropy) and apply gradients to variables. It will be done by an optimizer: GradientDescent or Adagrad. 

#### Define prediction
Do you want to know how many of the cases in a mini-batch has been classified correctly? lets count them.
#### Define accuracy
It makes more sense to report accuracy using average of correct cases.



```python

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

```

## traning the model


```python
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
```
