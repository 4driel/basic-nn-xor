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

As explained with the operation of a single artificial neuron. Forward propagation involves a weighted sumation of the inputs at every stage, and a squashing activation function. This would get repeated over and over depending of how many layers "deep" the network is.

Calculating these values is not as hard as it would seem. As, if representing all the operations as matrix operations, the computations can be made in bulk, on a per layer basis.

Using the first weigthed sumation as an example. The result could be given like this:

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/hidden-sum-expanded.jpg">
</p>
<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/hidden-sum.jpg">
</p>

Continuing with the explanation, after calculating the first weighted sumation, next is applying the activation function to s<sup>1</sup>. In this example, we'll use the logistic sigmoid function as our activation function.

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
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/numerical-y.jpg">
</p>

The full forward propagation would give these results:

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/numerical-s1.jpg">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/numerical-a1.jpg">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/numerical-a2.jpg">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/numerical-yhat.jpg">
</p>

The approximated values are nowhere near what they should be. This is because, as the initial weights were generated randomly, it was pretty improvable for the network to generate a good approximation on the first forward-propagation. This is what you call an "un-trained" nn. Until the weights are adjusted appropriately, the approximation will be poor.

To "train" a nn, a process called "back-propagation" is used. Back-propagation is also pretty self-explainatory, it propagates the computed error backwards, adjusting the weights accordingly, so on the next forward propagation, the error is smaller.

To explain the maths involved in back-propagation, we need to first define how we are going to calculate the error. A common way of doing it is the sum of squared deltas.

<p align="center">
  <img src="https://github.com/4driel/basic-nn-xor/blob/readme-edit/images/cost-function.jpg">
</p>

<p align="center">
To minimize the error, a gradient descent algorith is used, where the gradients are calculated by the cost function being partially differentiated with respect to each weight set. Looking at the gradient for w<sup>2</sup> first, it would go like this:
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\frac{\partial&space;E}{\partial&space;W^2}&space;&=&space;\frac{\frac12&space;\sum&space;(y&space;-&space;\hat{y})^2}{\partial&space;W^2}\\&space;&=&space;\frac12&space;\sum&space;\frac{\partial&space;(y&space;-&space;\hat{y})^2}{\partial&space;W^2}\\&space;&=&space;-\sum&space;(y-\hat{y})\frac{\partial&space;\hat{y}}{\partial&space;W^2}&space;\\&space;&=&space;-\sum&space;(y-\hat{y})\frac{\partial&space;s(s^2)}{\partial&space;W^2}&space;&&&space;\text{substitute&space;}&space;\hat{y}=s(s^2)\\&space;&=&space;-\sum&space;(y-\hat{y})&space;\frac{\partial&space;s(s^2)}{\partial&space;s^2}&space;\frac{\partial&space;s^2}{\partial&space;W^2}&space;&&&space;\text{apply&space;the&space;chain&space;rule}\\&space;\end{align*}" title="\begin{align*} \frac{\partial E}{\partial W^2} &= \frac{\frac12 \sum (y - \hat{y})^2}{\partial W^2}\\ &= \frac12 \sum \frac{\partial (y - \hat{y})^2}{\partial W^2}\\ &= -\sum (y-\hat{y})\frac{\partial \hat{y}}{\partial W^2} \\ &= -\sum (y-\hat{y})\frac{\partial s(s^2)}{\partial W^2} && \text{substitute } \hat{y}=s(s^2)\\ &= -\sum (y-\hat{y}) \frac{\partial s(s^2)}{\partial s^2} \frac{\partial s^2}{\partial W^2} && \text{apply the chain rule}\\ \end{align*}" />
</p>

To diferentiate <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;s(s^2))}{\partial&space;s^2}">, you simply calculate the derivative of our activation function.

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s'(z)&space;=&space;\frac{e^{-z}}{(1&space;&plus;&space;e^{-z})^2}">
</p>

And lastly, to differentiate <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;s^2}{\partial&space;W^2}"> you substitude <img src="https://latex.codecogs.com/gif.latex?\inline&space;s^2=a^1W^2"> and you get:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;W^2}=-\sum(y&space;-&space;\hat{y})s'(s^2)\frac{\partial&space;a^1W^2}{\partial&space;W^2}=-\sum(y&space;-&space;\hat{y})s'(s^2)\frac{\partial&space;a^1}{\partial&space;W^2}">
</p>

For w<sup>1</sup> it's the same procedure. There is only need to apply the chain rule and substututions further until you are able to diferentiate with respect to it.

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\frac{\partial&space;E}{\partial&space;W^1}&space;&=&space;-\frac12\sum\frac{\partial&space;(y&space;-&space;\hat{y})^2}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;\frac{\partial&space;\hat{y}}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;\frac{\partial&space;\hat{y}}{\partial&space;s^2}&space;\frac{\partial&space;s^2}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;\frac{\partial&space;s(s^2)}{\partial&space;s^2}&space;\frac{\partial&space;s^2}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;s'(s^2)&space;\frac{\partial&space;s^2}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;s'(s^2)&space;\frac{\partial&space;a^1W^2}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;s'(s^2)&space;W^2\frac{\partial&space;a^1}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;s'(s^2)&space;W^2\frac{\partial&space;s(s^1)}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;s'(s^2)&space;W^2\frac{\partial&space;s(s^1)}{\partial&space;s^1}\frac{\partial&space;s^1}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;s'(s^2)&space;W^2&space;s'(s^1)&space;\frac{\partial&space;s^1}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;s'(s^2)&space;W^2&space;s'(s^1)&space;\frac{\partial&space;xW^1}{\partial&space;W^1}\\&space;&=&space;-\sum(y&space;-&space;\hat{y})&space;s'(s^2)&space;W^2&space;s'(s^1)&space;x&space;\end{align*}" title="\tiny \begin{align*} \frac{\partial E}{\partial W^1} &= -\frac12\sum\frac{\partial (y - \hat{y})^2}{\partial W^1}\\ &= -\sum(y - \hat{y}) \frac{\partial \hat{y}}{\partial W^1}\\ &= -\sum(y - \hat{y}) \frac{\partial \hat{y}}{\partial s^2} \frac{\partial s^2}{\partial W^1}\\ &= -\sum(y - \hat{y}) \frac{\partial s(s^2)}{\partial s^2} \frac{\partial s^2}{\partial W^1}\\ &= -\sum(y - \hat{y}) s'(s^2) \frac{\partial s^2}{\partial W^1}\\ &= -\sum(y - \hat{y}) s'(s^2) \frac{\partial a^1W^2}{\partial W^1}\\ &= -\sum(y - \hat{y}) s'(s^2) W^2\frac{\partial a^1}{\partial W^1}\\ &= -\sum(y - \hat{y}) s'(s^2) W^2\frac{\partial s(s^1)}{\partial W^1}\\ &= -\sum(y - \hat{y}) s'(s^2) W^2\frac{\partial s(s^1)}{\partial s^1}\frac{\partial s^1}{\partial W^1}\\ &= -\sum(y - \hat{y}) s'(s^2) W^2 s'(s^1) \frac{\partial s^1}{\partial W^1}\\ &= -\sum(y - \hat{y}) s'(s^2) W^2 s'(s^1) \frac{\partial xW^1}{\partial W^1}\\ &= -\sum(y - \hat{y}) s'(s^2) W^2 s'(s^1) x \end{align*}" />
</p>
