<div align="center">
  
# Differentiable Drawing and Sketching

This repository contains the __official__ implementation of the paper ['Differentiable Drawing and Sketching.'](https://arxiv.org/abs/2103.16194)


<p align="center">
  <a href="https://arxiv.org/abs/2103.16194">ArXiv</a> •
  <a href="https://paperswithcode.com/paper/differentiable-drawing-and-sketching">Papers With Code</a> •
  <a href="#about">About</a> •
  <a href="#usage">Usage</a> •
  <a href="#experiments">Experiments</a> 
</p>

<img src="results/vancouver/final-800px.png"/>
</a>

</div>

## About

This is a bottom-up relaxed differentiable relaxation of point/line/curve rasterisation and sketch generation implemented entirely as GPU-accelerated tensor operations in [PyTorch](http://pytorch.org). We relax the act of rasterising primitive objects such that there is a gradient with respect to the primitive's parameters from every pixel in the image. Rasterisations of primitives can then be composed in different ways (so long as they are differentiable) to achieve a resultant output which can be optimised. 

We demonstrate that it is possible to perform gradient descent directly through the rasteriser to make the generated image match a target image according to some loss (which could be a simple function, or even a complex neural network -- the image above was created using the [LPIPS Perceptual Loss with a VGG16 network](https://github.com/richzhang/PerceptualSimilarity) to optimise 2000 straight line segments and their respective colours). We also demonstrate the use of the rasteriser in auto-encoder models that learn to both produce reconstructions whilst simultaneously learning the actual parameters of underlying vector primitives, and thus differentiably perform [automatic image tracing (aka autotracing)](https://en.wikipedia.org/wiki/Image_tracing).

## Usage

### Installation

Clone the source, then from the source directory install the dependencies (`pip -r requirements.txt`) and run `python setup.py install` to install the library and commandline tools.

### Demonstration Notebooks

The [samples/](samples/) directory contains a number of jupyter notebooks demonstrating how to implement different types of primitive parameterisations, as well as demonstrating direct optimisation of primitive parameters using gradient descent through the rasteriser.

### Tools

#### Image Optimisation

## Experiments

### Optimisation

### Auto-encoders




