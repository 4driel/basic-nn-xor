# A basic feed-forward nn approximating an xor gate

Hello all!

This is my first attempt at creating a neural network. I made the most basic one I could think of in order to get familiar with all the maths. Hopefully some of you find this example useful in your own endevours.

### Prequisites
- Basic multi-var calculus
- Linear algebra

### So what is this neural network you speak of?

A neural network is basically a function approximator that is crudely modeled on the interconnections neurons interface with inside our brains. That, through some funky maths, are able to approximate pretty much any continuous function.

<<<<<<< HEAD
As can seen on the images below, an antificial neuron bears a lot of resemblance to the biological one. The dendrites can be compared to the weighted synaptic inputs W1 ... Wn and the synaptic terminations to the output a1.
=======
<p align="center">
  <img src="https://cdn.rawgit.com/4driel/basic-nn-xor/master/images/biological-neuron.svg" height="200">
  <img src="https://cdn.rawgit.com/4driel/basic-nn-xor/master/images/artificial-neuron.svg" height="200">
</p>

### So what does an nn look like and how does it work?

A nn has an input layer, comprised of several input units depending on the size of your input vector; an output layer, comprised of one or more output units; and in some cases one or more hidden layers.

Your inputs  go through a weighted sum from the input layer into the next. The next layer applies a non-linear squashing function to it. And depending on how "deep" you nn is, this can be repeated serveral times until the result comes out of your output layer. This proccess is called forward-propagation.
>>>>>>> master

A neural "network" is just a bunch of artificial neurons connected together.

<p align="center">
  <img src="https://cdn.rawgit.com/4driel/basic-nn-xor/master/images/biological-neuron.svg" height="200">
  <img src="https://cdn.rawgit.com/4driel/basic-nn-xor/master/images/artificial-neuron.svg" height="200">
</p>
<p align="center">
  Image 1. Left: Simplified represetation of a biological neuron. Right: Diagram of artificial neuron.
</p>

### And how does it work?

In simplified terms, you map your inputs to your outputs by adjusting the synaptic weights. And that's it. Sounds easy right? Let's look at an example.

### Forward-propagation

The first process involved in the operation of an nn is called forward-propagation. Forward-propagation does what it sounds like, it propagates the input all the way through the network until you get an output result.

Lets look at our first nn and try to understand what maths the input goes through before emerging from the other side.

