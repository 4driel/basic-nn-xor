# A basic feed-forward nn approximating an xor gate

Hello all!

This is my first attempt at creating a neural network. I made the most basic one I could think of in order to get familiar with all the maths. Hopefully some of you find this example useful in your own endevours.

## Prequisites
- Basic multi-var calculus
- Linear algebra

## Introduction
So how does a neural network work?

You have your inputs, they get multiplied by their respective synaptic weights, then linearly summed and then squashed by an activation function. This process is called forward-propagation. This can be repeated a number of times depending on how "deep" you nn is.

Unless you had some astronomical luck when asigning your initial weights, you output will not match your desired output at first. This is because your nn is not yet "trained". The common aproach to train nn is gradient descent with back-propagation

First let's try forward-propagation.

## Forward-propagation


![Equation]()

## Notes
I used these resources when making this (in case you also find them useful):
- https://github.com/stephencwelch/Neural-Networks-Demystified
- http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
