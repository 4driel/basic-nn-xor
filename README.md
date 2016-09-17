# A basic feed-forward nn approximating an xor gate

Hello all!

This is my first attempt at creating a neural network. In order to get familiar with the maths involved, I made the most basic nn I could think of. An xor gate was the first thing that came to mind.

So let's look at how to create one. Hopefully some of you find this example useful in your own endevours.

### Prequisites
- Basic multi-var calculus
- Linear algebra

### So what is this neural network you speak of?

An artificial neural network is basically a function approximator that is crudely modeled on the way our brains are internally "wired". In the same way that biological neural networks have a bunch of neurons interconnected to each other, an ann is just a bunch of artificial neurons connected together.

The distributed way the computations inside an ann are done is what makes them such good approximators. It is said that even the most basic nns can approximate pretty much any continuos function.

To understand how a nn works, we first have to understand what a single artificial neuron does. If we observe the structure of an artificial neuron when compared to a bilogical one, we can spot a lot of similitudes. For instance, the synaptic connections labeled w<sub>1</sub> ... w<sub>n</sub> can be compared to dendrites. These synaptic connections multiply their inputs by a certain factor, so when all inputs are sumated, the degree in which certain inputs influence the final result of the neuron can be altered. 

The next section of the artificial neuron labeled f(s<sub>1</sub>) is what is called an activation function. Activation functions are non-linear squashing functions that enable nns to approximate complex non-linear functions. That is to say, if activation functions are not added to an nn, no mather how many layers the network has, it would not be able to approximate functions dealing with non linearly separable data.

<p align="center">
  <img src="https://cdn.rawgit.com/4driel/basic-nn-xor/master/images/biological-neuron.svg" height="200">
  <img src="https://cdn.rawgit.com/4driel/basic-nn-xor/master/images/artificial-neuron.svg" height="200">
</p>
<p align="center">
  Image 1. Left: Simplified representation of a biological neuron. Right: Diagram of artificial neuron.
</p>

### So how does it work then?

To explain what processes are involved in the operation of a neural network. Let's look at the example in hand, a simple 3 layer network approximating an XOR gate. An nn which is consisting of  2 input units, 3 hidden units, and 1 output unit.

<p align="center">
  <img src="https://cdn.rawgit.com/4driel/basic-nn-xor/master/images/neural-network.svg" height="200">
</p>
<p align="center">
  Image 2. Simple neural network (2 inputs, 3 hidden units, 1 output)
</p>

The first process is called forward-propagation. And it does basically what it says. It propagates the inputs through all the different layers in the nn until it reaches the output, then you get a result.

As explained with the operation of a single artificial neuron. Forward propagation involves a weighted sumation of the inputs at every stage, and a squashing activation function. This would get repeated over and over depending of how many layer "deep" the network is.

Calculating these values is not as hard as it would seem. As, if representing all the operations as matrix operations. The computation can be made in bulk, on a per layer basis.

Using the first weigthed sumation as an example. The result could be given like this:

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/hidden-sum-expanded.jpg">
</p>
<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/hidden-sum.jpg">
</p>

Continuing with the explanation, after calculating the first weighted sumation, next is applying the activation function to s<sup>1</sup>. In this example, well use the logistic sigmoid function as our activation function.

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/sigmoid.jpg">
</p>

So the hidden layer activity would be just:

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/hidden-activity.jpg">
</p>

As for the rest of forward-propagation, it would imply only repeating the same steps for the next layer.

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/output-sum.jpg">
</p>

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/output-result.jpg">
</p>

#### Numerical example

Let's take these initial weight matrices and input:

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/numerical-w1.jpg">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/numerical-w2.jpg">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/numerical-x.jpg">
</p>

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/numerical-s1.jpg">
</p>

[ 0.96043643  0.46803635  1.91756236]
 [-0.29696591  0.62271219  1.1089292 ]]
