# A basic feed-forward nn approximating an xor gate

Hello all!

This is my first attempt at creating a neural network. I made the most basic one I could think of in order to get familiar with all the maths. Hopefully some of you find this example useful in your own endevours.

## Prequisites
- Basic multi-var calculus
- Linear algebra

## Introduction
### So what is this neural network you speak of?

A neural network is basically a function approximator that is crudely modeled on the interconnections neurons interface with inside our brains. Through some funky maths, they are able to approximate pretty much any continuous function.

### So what does an nn look like and how does it work?

A nn has an input layer, comprised of several input units depending on the size of your input vector; an output layer, comprised of one or more output units; and in some cases one or more hidden layers.

Your inputs  go through a weighted sum from the input layer into the next. The next layer applies a non-linear squashing function to it. And depending on how "deep" you nn is, this can be repeated serveral times until the result comes out of your output layer. This proccess is called forward-propagation.

Unless you had some astronomical good luck when asigning your initial weights, you output will not match your desired output after doing the first forward-propagation. This is because your nn is not yet "trained". The common aproach to training a nn is gradient descent with back-propagation. Back-propagation involves calculating a gradient in terms of how much your output result is affected by each weight, and then adjusting the weights in order to minimize the error. This is where calculus comes into play.

So, in simplified terms, you map your inputs to your outputs by adjusting the synaptic weights. And that's it.

Let's look at the maths involved in forward and back-propagation.

<img src="https://github.com/4driel/basic-nn-xor/blob/master/images/nn.png" width="300">

_Image 1. Basic neural network with 2 input units, 3 hidden units and 1 output unit._

- x: input vector
- W1, W2: synaptic weight sets
- h: hidden layer
- s1, s2: sumation of linear functions from the previous layer
- a1, y: result from applying a non-linear squashing function to s1, s2

## Forward-propagation

<img src="http://www.sciweavers.org/tex2img.php?eq=s_1%3Dx%5Ccdot%20W_1&bc=White&fc=Black&im=jpg&fs=12&ff=modern&edit=0" align="center" border="0" alt="s_1=x\cdot W_1" width="81" height="17" />


## Notes
I used these resources when making this (in case you also find them useful):
- https://github.com/stephencwelch/Neural-Networks-Demystified
- http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
