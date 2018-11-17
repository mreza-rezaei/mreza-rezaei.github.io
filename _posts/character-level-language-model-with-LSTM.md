---
published: true
---

## Character Level Language Model With LSTM And Simulation In Tensorflow 

I was so confused when i start Recurrent Neural Networks [Rnns](https://en.wikipedia.org/wiki/Recurrent_neural_network) and languge modelling , in this post i'm going to create a charater model laguage   and simulate that with tensorflow platform

## learning Goals 
- [Recurrent Networks]
    - [LSTM RNN ]	
- [Character Langauge Model]
- [Create Character Language Model In Tensorflow]

## Recurrent Networks

Up until now, all of the networks that we've learned and worked with really have no sense of time. They are static. They cannot remember sequences, nor can they understand order outside of the spatial dimensions we offer it. Imagine for instance that we wanted a network capable of reading. As input, it is given one letter at a time. So let's say it were given the letters 'n', 'e', 't', 'w', 'o', 'r', and we wanted it to learn to output 'k'. It would need to be able to reason about inputs it received before the last one it received, the letters before 'r'. But it's not just letters.
Consider the way we look at the world. We don't simply download a high resolution image of the world in front of us. We move our eyes. Each fixation takes in new information and each of these together in sequence help us perceive and act. That again is a sequential process.
Recurrent neural networks let us reason about information over multiple timesteps. They are able to encode what it has seen in the past as if it has a memory of its own. It does this by basically creating one HUGE network that expands over time. It can reason about the current timestep by conditioning on what it has already seen. By giving it many sequences as batches, it can learn a distribution over sequences which can model the current timestep given the previous timesteps. But in order for this to be practical, we specify at each timestep, or each time it views an input, that the weights in each new timestep cannot change. We also include a new matrix, H, which reasons about the past timestep, connecting each new timestep. For this reason, we can just think of recurrent networks as ones with loops in it.

## LSTM RNN

The mechanics of LSTM,  it uses a combinations of gating cells to control its contents and by having gates, it is able to block the flow of the gradient, avoiding too many multiplications during backprop.  For more details, I highly recommend reading: https://colah.github.io/posts/2015-08-Understanding-LSTMs/.

## Character Langauge Model

best explation i find in character level language model decription in web is [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
in this blog you can understand Rnns and how its work in character level language modeling.

in this post i'm goinig to create character language modeling with [tensorflow ](https://www.tensorflow.org/)


lets start this !!!

## import tensorflow and load some text file 
you can download used text from [here](http://cs.stanford.edu/people/karpathy/char-rnn/ web page)


```python
%pylab
import tensorflow as tf
from six.moves import urllib
import ssl

with open('warpeace_input.txt', 'r') as fp:
    txt = fp.read()
```


```python
vocab = list(set(txt))
len(txt), len(vocab)
```

as you can see we have about 320 thousand characters and 83 unique characters in our vocabulary which we can use to help us train a model of language. 

let's look at characters befor get hand's dirty with that 


```python
vocab
```

We'll just clean up the text a little. This isn't necessary, but can help the training along a little. In the example text I provided, there is a lot of white space (those \t's are tabs). I'll remove them. There are also repetitions of \n, new lines, which are not necessary. The code below will remove the tabs, ending whitespace, and any repeating newlines. Replace this with any preprocessing that makes sense for your dataset. Try to boil it down to just the possible letters for what you want to learn/synthesize while retaining any meaningful patterns:


```python
txt = "\n".join([txt_i.strip()
                 for txt_i in txt.replace('\t', '').split('\n')
                 if len(txt_i)])
```

And then create a mapping which can take us from the letter to an integer look up table of that letter (and vice-versa).  To do this, we'll use an `OrderedDict` from the `collections` library.  In Python 3.6, this is the default behavior of dict, but in earlier versions of Python, we'll need to be explicit by using OrderedDict.


```python
from collections import OrderedDict

encoder = OrderedDict(zip(vocab, range(len(vocab))))
decoder = OrderedDict(zip(range(len(vocab)), vocab))
```

We'll store a few variables that will determine the size of our network.  First, `batch_size` determines how many sequences at a time we'll train on.  The `seqence_length` parameter defines the maximum length to unroll our recurrent network for.  This is effectively the depth of our network during training to help guide gradients along.  Within each layer, we'll have `n_cell` LSTM units, and `n_layers` layers worth of LSTM units.  Finally, we'll store the total number of possible characters in our data, which will determine the size of our one hot encoding (like we had for MNIST in Session 3).


```python
# Number of sequences in a mini batch
batch_size = 100

# Number of characters in a sequence
sequence_length = 100

# Number of cells in our LSTM layer
n_cells = 256

# Number of LSTM layers
n_layers = 2

# Total number of characters in the one-hot encoding
n_chars = len(vocab)
```

Let's now create the input and output to our network.  We'll use placeholders and feed these in later.  The size of these need to be [`batch_size`, `sequence_length`].  We'll then see how to build the network in between.




```python
X = tf.placeholder(tf.int32, [batch_size, sequence_length], name='X')

# We'll have a placeholder for our true outputs
Y = tf.placeholder(tf.int32, [batch_size, sequence_length], name='Y')
```

The first thing we need to do is convert each of our `sequence_length` vectors in our batch to `n_cells` LSTM cells.  We use a lookup table to find the value in `X` and use this as the input to `n_cells` LSTM cells.  Our lookup table has `n_chars` possible elements and connects each character to `n_cells` cells.  We create our lookup table using `tf.get_variable` and then the function `tf.nn.embedding_lookup` to connect our `X` placeholder to `n_cells` number of neurons.


```python
# we first create a variable to take us from our one-hot representation to our LSTM cells
embedding = tf.get_variable("embedding", [n_chars, n_cells])

# And then use tensorflow's embedding lookup to look up the ids in X
Xs = tf.nn.embedding_lookup(embedding, X)

# The resulting lookups are concatenated into a dense tensor
print(Xs.get_shape().as_list())
```

Recurrent neural networks share their weights across timesteps.  So we don't want to have one large matrix with every timestep, but instead separate them.  We'll use `tf.split` to split our `[batch_size, sequence_length, n_cells]` array in `Xs` into a list of `sequence_length` elements each composed of `[batch_size, n_cells]` arrays.  This gives us `sequence_length` number of arrays of `[batch_size, 1, n_cells]`.  We then use `tf.squeeze` to remove the 1st index corresponding to the singleton `sequence_length` index, resulting in simply `[batch_size, n_cells]`.


```python
# Let's create a name scope for the operations to clean things up in our graph
with tf.name_scope('reslice'):
    Xs = [tf.squeeze(seq, [1])
          for seq in tf.split(Xs, sequence_length, axis=1)]
```

With each of our timesteps split up, we can now connect them to a set of LSTM recurrent cells.  We tell the `tf.contrib.rnn.BasicLSTMCell` method how many cells we want, i.e. how many neurons there are, and we also specify that our state will be stored as a tuple.  This state defines the internal state of the cells as well as the connection from the previous timestep.  We can also pass a value for the `forget_bias`.  Be sure to experiment with this parameter as it can significantly effect performance (e.g. Gers, Felix A, Schmidhuber, Jurgen, and Cummins, Fred. Learning to forget: Continual prediction with lstm. Neural computation, 12(10):2451–2471, 2000).


```python
cells = tf.contrib.rnn.BasicLSTMCell(num_units=n_cells)
```

Let's take a look at the cell's state size:


```python
cells.state_size
```

`c` defines the internal memory and `h` the output.  We'll have as part of our `cells`, both an `initial_state` and a `final_state`.  These will become important during inference and we'll see how these work more then.  For now, we'll set the `initial_state` to all zeros using the convenience function provided inside our `cells` object, `zero_state`:


```python
initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
```

Looking at what this does, we can see that it creates a `tf.Tensor` of zeros for our `c` and `h` states for each of our `n_cells` and stores this as a tuple inside the `LSTMStateTuple` object:


```python
initial_state
```

create deeper Recurrent network


```python
# Build deeper recurrent net if using more than 1 layer
if n_layers > 1:
    cells = [cells]
    for layer_i in range(1, n_layers):
        with tf.variable_scope('{}'.format(layer_i)):
            this_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=n_cells)
            cells.append(this_cell)
    cells = tf.contrib.rnn.MultiRNNCell(cells)
    initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)
```


```python
initial_state

```

So far, we haven't connected our recurrent cells to anything.  Let's do this now using the `tf.contrib.rnn.static_rnn` method.  We also pass it our `initial_state` variables.  It gives us the `outputs` of the rnn, as well as their states after having been computed.  Contrast that with the `initial_state`, which set the LSTM cells to zeros.  After having computed something, the cells will all have a different value somehow reflecting the temporal dynamics and expectations of the next input.  These will be stored in the `state` tensors for each of our LSTM layers inside a `LSTMStateTuple` just like the `initial_state` variable.


```python
# this will return us a list of outputs of every element in our sequence.
# Each output is `batch_size` x `n_cells` of output.
# It will also return the state as a tuple of the n_cells's memory and
# their output to connect to the time we use the recurrent layer.
outputs, state = tf.contrib.rnn.static_rnn(cells, Xs, initial_state=initial_state)

# We'll now stack all our outputs for every cell
outputs_flat = tf.reshape(tf.concat(outputs, axis=1), [-1, n_cells])
```

We now create a softmax layer just like we did in Session 3 and in Session 3's homework.  We multiply our final LSTM layer's `n_cells` outputs by a weight matrix to give us `n_chars` outputs.  We then scale this output using a `tf.nn.softmax` layer so that they become a probability by exponentially scaling its value and dividing by its sum.  We store the softmax probabilities in `probs` as well as keep track of the maximum index in `Y_pred`:


```python
with tf.variable_scope('prediction'):
    W = tf.get_variable(
        "W",
        shape=[n_cells, n_chars],
        initializer=tf.random_normal_initializer(stddev=0.1))
    b = tf.get_variable(
        "b",
        shape=[n_chars],
        initializer=tf.random_normal_initializer(stddev=0.1))

    # Find the output prediction of every single character in our minibatch
    # we denote the pre-activation prediction, logits.
    logits = tf.matmul(outputs_flat, W) + b

    # We get the probabilistic version by calculating the softmax of this
    probs = tf.nn.softmax(logits)

    # And then we can find the index of maximum probability
    Y_pred = tf.argmax(probs, axis=1)
```


To train the network, we'll measure the loss between our predicted outputs and true outputs. We could use the probs variable, but we can also make use of tf.nn.softmax_cross_entropy_with_logits which will compute the softmax for us. We therefore need to pass in the variable just before the softmax layer, denoted as logits (unscaled values). This takes our variable logits, the unscaled predicted outputs, as well as our true outputs, Y. Before we give it Y, we'll need to reshape our true outputs in the same way, [batch_size x timesteps, n_chars]. Luckily, tensorflow provides a convenience for doing this, the tf.nn.sparse_softmax_cross_entropy_with_logits function:


```python
tf.nn.sparse_softmax_cross_entropy_with_logits?
```


```python
with tf.variable_scope('loss'):
    # Compute mean cross entropy loss for each output.
    Y_true_flat = tf.reshape(tf.concat(Y, axis=1), [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_true_flat,logits=logits)
    mean_loss = tf.reduce_mean(loss)
```

Finally, we can create an optimizer in much the same way as we've done with every other network.  Except, we will also "clip" the gradients of every trainable parameter.  This is a hacky way to ensure that the gradients do not grow too large (the literature calls this the "exploding gradient problem").  However, note that the LSTM is built to help ensure this does not happen by allowing the gradient to be "gated".  To learn more about this, please consider reading the following material:

http://www.felixgers.de/papers/phd.pdf  
https://colah.github.io/posts/2015-08-Understanding-LSTMs/


```python
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    gradients = []
    clip = tf.constant(5.0, name="clip")
    for grad, var in optimizer.compute_gradients(mean_loss):
        gradients.append((tf.clip_by_value(grad, -clip, clip), var))
    updates = optimizer.apply_gradients(gradients)
```

Let's take a look at the graph:

Below is the rest of code we'll need to train the network.  I do not recommend running this inside Jupyter Notebook for the entire length of the training because the network can take 1-2 days at least to train, and your browser may very likely complain.  Instead, you should write a python script containing the necessary bits of code and run it using the Terminal.  We didn't go over how to do this, so I'll leave it for you as an exercise.  The next part of this notebook will have you load a pre-trained network.


```python
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

cursor = 0
it_i = 0
while True:
    Xs, Ys = [], []
    for batch_i in range(batch_size):
        if (cursor + sequence_length) >= len(txt) - sequence_length - 1:
            cursor = 0
        Xs.append([encoder[ch]
                   for ch in txt[cursor:cursor + sequence_length]])
        Ys.append([encoder[ch]
                   for ch in txt[cursor + 1: cursor + sequence_length + 1]])

        cursor = (cursor + sequence_length)
    Xs = np.array(Xs).astype(np.int32)
    Ys = np.array(Ys).astype(np.int32)

    loss_val, _ = sess.run([mean_loss, updates],
                           feed_dict={X: Xs, Y: Ys})
    print(it_i, loss_val)

    if it_i % 500 == 0:
        p = sess.run([Y_pred], feed_dict={X: Xs})[0]
        preds = [decoder[p_i] for p_i in p]
        print("".join(preds).split('\n'))

    it_i += 1
```
