# Differential-Privacy-Federated-Learning

Differential Privacy applied to distributed learning NN with PyTorch. 
A pure python introduction.
This is a work in progress but my aim is to show and discuss examples of usage from frameworks such as PySyft while 
discussing implementation details and revising NeuralNet basics.

Based on the work of https://github.com/gitgik.

## Introduction
If you are new to NN, I recommend the following videos to give the proper intuitions.
- [Neural Net Intro](https://www.youtube.com/watch?v=aircAruvnKk&t)
- [Learning](https://www.youtube.com/watch?v=IHZwWFHWa-w) 
- [Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [Backpropagation 2](https://www.youtube.com/watch?v=tIeHLnjs5U8)

Also, I recommend checking this amazing channel [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw).

## Tensors intro:
Work in progress

## Creating simple Neural Networks:
- [Low Level Single Layer NN](single_layer_test.py)
- [Multilayer NN as a Class](torch_nn_1foward_class.py) With Pytorch wrappers
- [Multilayer NN](torch_nn_1foward.py) With high level Pytorch wrappers

## Diferential Privacy
- [Introduction](diferential_priv_intro.py)

## PySyft Neural Net
- [PySyft intro](diferential_priv_pysyft_intro.py)
- [PySyft Distributed MNIST NN](diferential_priv_pysyft_mnist.py)
