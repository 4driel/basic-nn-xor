# A basic feed-forward nn approximating an xor gate

Hello all!

This is my first attempt at creating a neural network. I made the most basic one I could think of in order to get familiar with all the maths. Hopefully some of you find this example useful in your own endevours.

### Prequisites
- Basic multi-var calculus
- Linear algebra

### So what is this neural network you speak of?

A neural network is basically a function approximator that is crudely modeled on the interconnections neurons interface with inside our brains. That, through some funky maths, are able to approximate pretty much any continuous function.

As can seen on the images below, an antificial neuron bears a lot of reselmbance to the biological one. The dendrites could be compared to the weighted synaptic inputs $$ W_1 ... W_n $$ and the synaptic terminations could be compared to the result $$ a_1 $$.

A neural network is just a bunch of artificial neurons connected together.

<p align="center">
  <img src="https://cdn.rawgit.com/4driel/basic-nn-xor/master/images/biological-neuron.svg" height="200">
  <img src="https://cdn.rawgit.com/4driel/basic-nn-xor/master/images/artificial-neuron.svg" height="200">
</p>
<p align="center">
  Image 1. Left: Simplified represetation of a biological neuron. Right: Diagram of artificial neuron.
</p>

### And how does it work?

In simplified terms, you map your inputs to your outputs by adjusting the synaptic weights. And that's it.
