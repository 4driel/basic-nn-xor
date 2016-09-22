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

To understand how a nn works, we first have to understand what a single artificial neuron does. If we observe the structure of an artificial neuron when compared to a bilogical one, we can spot a lot of similitudes. For instance, the synaptic connections labeled <img src="https://latex.codecogs.com/gif.latex?\inline&space;W_1&space;...&space;W_n" title="W_1 ... W_n" /> can be compared to dendrites. These synaptic connections multiply their inputs by a certain factor, so when all inputs are sumated, the degree in which certain inputs influence the final result of the neuron can be altered. 

The next section of the artificial neuron labeled <img src="https://latex.codecogs.com/gif.latex?\inline&space;f(s_1)" title="f(s_1)" /> is what is called an activation function. Activation functions are non-linear squashing functions that enable nns to approximate complex non-linear functions. That is to say, if activation functions are not added to an nn, no mather how many layers the network has, it would not be able to approximate functions dealing with non linearly separable data.

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
  <img src="https://latex.codecogs.com/gif.latex?\begin{smallmatrix}&space;x&space;\\&space;\begin{bmatrix}&space;x^1&space;&&space;x^2&space;\end{bmatrix}&space;\end{smallmatrix}&space;\begin{smallmatrix}&space;W^1\\&space;\begin{bmatrix}&space;w^1_{1,1}&space;&&space;w^1_{1,2}&space;&&space;w^1_{1,3}&space;\\&space;w^1_{2,1}&space;&&space;w^1_{2,2}&space;&&space;w^1_{2,3}&space;\end{bmatrix}&space;\end{smallmatrix}=&space;\begin{smallmatrix}&space;s^1\\&space;\begin{bmatrix}&space;x^1w^1_{1,1}&space;&plus;&space;x^2w^1_{2,1}&space;&&space;x^1w^1_{1,2}&space;&plus;&space;x^2w^1_{2,2}&space;&&space;x^2w^1_{1,3}&space;&plus;&space;x^2w^1_{2,3}&space;\end{bmatrix}&space;\end{smallmatrix}" title="\begin{smallmatrix} x \\ \begin{bmatrix} x^1 & x^2 \end{bmatrix} \end{smallmatrix} \begin{smallmatrix} W^1\\ \begin{bmatrix} w^1_{1,1} & w^1_{1,2} & w^1_{1,3} \\ w^1_{2,1} & w^1_{2,2} & w^1_{2,3} \end{bmatrix} \end{smallmatrix}= \begin{smallmatrix} s^1\\ \begin{bmatrix} x^1w^1_{1,1} + x^2w^1_{2,1} & x^1w^1_{1,2} + x^2w^1_{2,2} & x^2w^1_{1,3} + x^2w^1_{2,3} \end{bmatrix} \end{smallmatrix}" />
</p>
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?s^1&space;=&space;xW^1" title="s^1 = xW^1" />
</p>

Continuing with the explanation, after calculating the first weighted sumation, next is applying the activation function to <img src="https://latex.codecogs.com/gif.latex?\inline&space;s^1" title="s^1" />. In this example, we'll use the logistic sigmoid function as our activation function.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?s(z)=&space;\frac{1}{1&plus;e^{-z}}" title="s(z)= \frac{1}{1+e^{-z}}" />
</p>

So the hidden layer activity would be just:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?a^1=&space;s(s^1)" title="a^1= s(s^1)" />
</p>

As for the rest of forward-propagation, it would imply only repeating the same steps for the next layer.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?s^2&space;=&space;a^1W^2" title="s^2 = a^1W^2" />
</p>
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\hat{y}=&space;s(s^2)" title="\hat{y}= s(s^2)" />
</p>

#### Numerical example

Let's take these initial weight matrices and input:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;&W^1=&space;\begin{bmatrix}&space;0.96&space;&&space;0.46&space;&&space;1.91&space;\\&space;-0.29&space;&&space;0.62&space;&&space;1.10&space;\end{bmatrix},&space;&&W^2=&space;\begin{bmatrix}&space;-0.96&space;\\&space;-0.13&space;\\&space;-0.74&space;\end{bmatrix},&space;&&&x=&space;\begin{bmatrix}&space;0&space;&&space;0&space;\\&space;0&space;&&space;1&space;\\&space;1&space;&&space;0&space;\\&space;1&space;&&space;1&space;\end{bmatrix},&space;&&&&y&space;=&space;\begin{bmatrix}&space;0&space;\\&space;1&space;\\&space;1&space;\\&space;0&space;\end{bmatrix}&space;\end{align*}" title="\begin{align*} &W^1= \begin{bmatrix}0.96&0.46&1.91\\-0.29&0.62&1.10\end{bmatrix}, &&W^2= \begin{bmatrix}-0.96\\-0.13\\-0.74\end{bmatrix}, &&&x= \begin{bmatrix}0&0\\0&1\\1&0\\1&1\end{bmatrix}, &&&&y= \begin{bmatrix}0\\1\\1\\0\end{bmatrix} \end{align*}" />
</p>

The full forward propagation would give these results:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;&s^1=&space;\begin{bmatrix}0&0&0\\-0.29&0.62&1.1\\0.96&0.46&1.91\\0.67&1.08&3.01\end{bmatrix},&space;&&a^1=\begin{bmatrix}0.5&0.5&0.5\\0.42800387&0.65021855&0.75026011\\0.72312181&0.61301418&0.87101915\\0.66150316&0.74649398&0.95302385\end{bmatrix},&space;&&&s^2=&space;\begin{bmatrix}0.915\\-1.0506046\\-1.41844295\\-1.4373249\end{bmatrix},&space;&&&&\hat{y}&space;=&space;\begin{bmatrix}0.28597777\\0.25910902\\0.1949058\\0.19195994\end{bmatrix}&space;\end{align*}" title="\begin{align*} &s^1= \begin{bmatrix}0&0&0\\-0.29&0.62&1.1\\0.96&0.46&1.91\\0.67&1.08&3.01\end{bmatrix}, &&a^1=\begin{bmatrix}0.5&0.5&0.5\\0.42800387&0.65021855&0.75026011\\0.72312181&0.61301418&0.87101915\\0.66150316&0.74649398&0.95302385\end{bmatrix}, &&&s^2= \begin{bmatrix}0.915\\-1.0506046\\-1.41844295\\-1.4373249\end{bmatrix}, &&&&\hat{y} = \begin{bmatrix}0.28597777\\0.25910902\\0.1949058\\0.19195994\end{bmatrix} \end{align*}" />
</p>

The approximated values are nowhere near what they should be. This is because, as the initial weights were generated randomly, it was pretty improvable for the network to generate a good approximation on the first forward-propagation. This is what you call an "un-trained" nn. Until the weights are adjusted appropriately, the approximation will be poor.

To "train" a nn, a process called "back-propagation" is used. Back-propagation is also pretty self-explainatory, it propagates the computed error backwards, adjusting the weights accordingly, so on the next forward propagation, the error is smaller.

To explain the maths involved in back-propagation, we need to first define how we are going to calculate the error. A common way of doing it is the sum of squared deltas.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?E=\frac12\sum(y-\hat{y})^2" title="E=\frac12\sum(y-\hat{y})^2" />
</p>

To minimize the error, a gradient descent algorith is used, where the gradients are calculated by the cost function being partially differentiated with respect to each weight set. Looking at the gradient for <img src="https://latex.codecogs.com/gif.latex?\inline&space;W^2" title="W^2" /> first, it would go like this:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{\partial&space;E}{\partial&space;W^2}&=\frac{\frac12\sum(y-\hat{y})^2}{\partial&space;W^2}\\&=\frac12\sum\frac{\partial(y-\hat{y})^2}{\partial&space;W^2}\\&=-\sum(y-\hat{y})\frac{\partial\hat{y}}{\partial&space;W^2}\\&=-\sum(y-\hat{y})\frac{\partial&space;s(s^2)}{\partial&space;W^2}&&\text{substitute&space;}\hat{y}=s(s^2)\\&=-\sum(y-\hat{y})\frac{\partial&space;s(s^2)}{\partial&space;s^2}\frac{\partial&space;s^2}{\partial&space;W^2}&&\text{apply&space;the&space;chain&space;rule}\\\end{align*}" title="\begin{align*}\frac{\partial E}{\partial W^2}&=\frac{\frac12\sum(y-\hat{y})^2}{\partial W^2}\\&=\frac12\sum\frac{\partial(y-\hat{y})^2}{\partial W^2}\\&=-\sum(y-\hat{y})\frac{\partial\hat{y}}{\partial W^2}\\&=-\sum(y-\hat{y})\frac{\partial s(s^2)}{\partial W^2}&&\text{substitute }\hat{y}=s(s^2)\\&=-\sum(y-\hat{y})\frac{\partial s(s^2)}{\partial s^2}\frac{\partial s^2}{\partial W^2}&&\text{apply the chain rule}\\\end{align*}" />
</p>

To diferentiate <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;s(s^2)}{\partial&space;W^2}" title="\frac{\partial s(s^2)}{\partial W^2}" />, you simply calculate the derivative of our activation function.

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?s'(z)&space;=&space;\frac{e^{-z}}{(1&space;&plus;&space;e^{-z})^2}">
</p>

And lastly, to differentiate <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;s^2}{\partial&space;W^2}"> you substitude <img src="https://latex.codecogs.com/gif.latex?\inline&space;s^2=a^1W^2"> and you get:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{\partial&space;E}{\partial&space;W^2}&=-\sum(y-\hat{y})s'(s^2)\frac{\partial&space;a^1&space;W^2}{\partial&space;W^2}\\&=-\sum(y-\hat{y})s'(s^2)a^1\end{align*}" title="\begin{align*}\frac{\partial E}{\partial W^2}&=-\sum(y-\hat{y})s'(s^2)\frac{\partial a^1 W^2}{\partial W^2}\\&=-\sum(y-\hat{y})s'(s^2)a^1\end{align*}" />
</p>

For <img src="https://latex.codecogs.com/gif.latex?\inline&space;W^1" title="W^1" /> it's the same procedure. There is only need to apply the chain rule and substitutions further until you are able to diferentiate with respect to it.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{\partial&space;E}{\partial&space;W^1}&=-\frac12\sum\frac{\partial(y-\hat{y})^2}{\partial&space;W^1}\\&=-\sum(y-\hat{y})\frac{\partial\hat{y}}{\partial&space;W^1}\\&=-\sum(y-\hat{y})\frac{\partial\hat{y}}{\partial&space;s^2}\frac{\partial&space;s^2}{\partial&space;W^1}\\&=-\sum(y-\hat{y})\frac{\partial&space;s(s^2)}{\partial&space;s^2}\frac{\partial&space;s^2}{\partial&space;W^1}\\&=-\sum(y-\hat{y})s'(s^2)\frac{\partial&space;s^2}{\partial&space;W^1}\\&=-\sum(y-\hat{y})s'(s^2)\frac{\partial&space;a^1W^2}{\partial&space;W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2\frac{\partial&space;a^1}{\partial&space;W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2\frac{\partial&space;s(s^1)}{\partial&space;W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2\frac{\partial&space;s(s^1)}{\partial&space;s^1}\frac{\partial&space;s^1}{\partial&space;W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2s'(s^1)\frac{\partial&space;s^1}{\partial&space;W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2s'(s^1)\frac{\partial&space;xW^1}{\partial&space;W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2s'(s^1)x\end{align*}" title="\begin{align*}\frac{\partial E}{\partial W^1}&=-\frac12\sum\frac{\partial(y-\hat{y})^2}{\partial W^1}\\&=-\sum(y-\hat{y})\frac{\partial\hat{y}}{\partial W^1}\\&=-\sum(y-\hat{y})\frac{\partial\hat{y}}{\partial s^2}\frac{\partial s^2}{\partial W^1}\\&=-\sum(y-\hat{y})\frac{\partial s(s^2)}{\partial s^2}\frac{\partial s^2}{\partial W^1}\\&=-\sum(y-\hat{y})s'(s^2)\frac{\partial s^2}{\partial W^1}\\&=-\sum(y-\hat{y})s'(s^2)\frac{\partial a^1W^2}{\partial W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2\frac{\partial a^1}{\partial W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2\frac{\partial s(s^1)}{\partial W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2\frac{\partial s(s^1)}{\partial s^1}\frac{\partial s^1}{\partial W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2s'(s^1)\frac{\partial s^1}{\partial W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2s'(s^1)\frac{\partial xW^1}{\partial W^1}\\&=-\sum(y-\hat{y})s'(s^2)W^2s'(s^1)x\end{align*}" />
</p>

As can be seen from the formulas for <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;E}{\partial&space;W^2}" title="\frac{\partial E}{\partial W^2}" /> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;E}{\partial&space;W^2}" title="\frac{\partial E}{\partial W^2}" />, the first couple of factors are the same. Something similar will happen when you do a deeper network. This can be exploited in order to speed computation time considerably.

#### Numerical example

Continuing with the same example, and taking the previous forward propagation results, the first backward would give these results:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;&s^1=&space;\begin{bmatrix}0&0&0\\-0.29&0.62&1.1\\0.96&0.46&1.91\\0.67&1.08&3.01\end{bmatrix},&space;&&a^1=\begin{bmatrix}0.5&0.5&0.5\\0.42800387&0.65021855&0.75026011\\0.72312181&0.61301418&0.87101915\\0.66150316&0.74649398&0.95302385\end{bmatrix},&space;&&&s^2=&space;\begin{bmatrix}0.915\\-1.0506046\\-1.41844295\\-1.4373249\end{bmatrix},&space;&&&&\hat{y}&space;=&space;\begin{bmatrix}0.28597777\\0.25910902\\0.1949058\\0.19195994\end{bmatrix}&space;\end{align*}" title="\begin{align*} &s^1= \begin{bmatrix}0&0&0\\-0.29&0.62&1.1\\0.96&0.46&1.91\\0.67&1.08&3.01\end{bmatrix}, &&a^1=\begin{bmatrix}0.5&0.5&0.5\\0.42800387&0.65021855&0.75026011\\0.72312181&0.61301418&0.87101915\\0.66150316&0.74649398&0.95302385\end{bmatrix}, &&&s^2= \begin{bmatrix}0.915\\-1.0506046\\-1.41844295\\-1.4373249\end{bmatrix}, &&&&\hat{y} = \begin{bmatrix}0.28597777\\0.25910902\\0.1949058\\0.19195994\end{bmatrix} \end{align*}" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{align*}&\frac{\partial&space;E}{\partial&space;W^2}=\begin{bmatrix}-0.10333551\\-0.11850021\\-0.1591743\end{bmatrix},&&\frac{\partial&space;E}{\partial&space;W^1}=\begin{bmatrix}0.01788182&0.00316357&0.00951631\\0.02702697&0.00347273&0.01873428\end{bmatrix}\end{align*}" title="\begin{align*}&\frac{\partial E}{\partial W^2}=\begin{bmatrix}-0.10333551\\-0.11850021\\-0.1591743\end{bmatrix},&&\frac{\partial E}{\partial W^1}=\begin{bmatrix}0.01788182&0.00316357&0.00951631\\0.02702697&0.00347273&0.01873428\end{bmatrix}\end{align*}" />
</p>

And running the forward propagation would yield this new result:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\hat{y}=\begin{bmatrix}0.32640392\\0.30955745\\0.24439711\\0.24612685\end{bmatrix}" title="\hat{y}=\begin{bmatrix}0.32640392\\0.30955745\\0.24439711\\0.24612685\end{bmatrix}" />
</p>

Still crappy results. Actually worse. This can happen when the nth dimensional valley traversed by the descent is not perfectly in the shape of a parabola. The error can increase temporarily for a couple of iterations. But if we run our nn with an arbitrarily large number of iterations. Lets say 7000, these are the results we get:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\begin{align*}&\hat{y}_\text{run&space;1}=\begin{bmatrix}0.02729357\\0.95632996\\0.9637799\\0.0350348\end{bmatrix},&&\hat{y}_\text{run&space;2}=\begin{bmatrix}0.03874082\\0.97634812\\0.97637353\\0.00810983\end{bmatrix},&&&\hat{y}_\text{run&space;3}=\begin{bmatrix}0.02783607\\0.95603128\\0.963397\\0.03510539\end{bmatrix}\end{align*}" title="\begin{align*}&\hat{y}_\text{run 1}=\begin{bmatrix}0.02729357\\0.95632996\\0.9637799\\0.0350348\end{bmatrix},&&\hat{y}_\text{run 2}=\begin{bmatrix}0.03874082\\0.97634812\\0.97637353\\0.00810983\end{bmatrix},&&&\hat{y}_\text{run 3}=\begin{bmatrix}0.02783607\\0.95603128\\0.963397\\0.03510539\end{bmatrix}\end{align*}" />
</p>

Pretty great, right?

NOTE: I'll adress overfitting on a later date.

I used these resources when trying to understand nns, maybe you'll also find them useful:
- http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
- https://github.com/stephencwelch/Neural-Networks-Demystified
