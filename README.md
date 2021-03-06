# Effective dimension in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Guigui14460/effective-dimension-pytorch/blob/master/examples/notebooks/effective_dimension.ipynb)

Implementation test of the [Abbas' et al] paper available on [arXiv](https://arxiv.org/abs/2112.04807). The initial implementation is available at [amyami187/effective_dimension](https://github.com/amyami187/effective_dimension). The goal of this metric is to estimate the size of that model occupies in the model's parameters space and checks the generalisation error of the evaluated model.

About the **local** effective dimension (go check the implementation at [amyami187/local_effective_dimension](https://github.com/amyami187/local_effective_dimension)), its goal is the same as the global effective dimension but it only analyzes the training result and not the large number of parameters set and can be applied to Machine Learning models too.

**:warning: Important note :** when the effective dimension is printed, more it is, better is the model generalization.

The [NNGeometry](https://nngeometry.readthedocs.io/en/latest/) package allow to efficiently compute the [Fisher information matrix](https://www.wikiwand.com/en/Fisher_information) with a [Kronecker-factored approximation](https://arxiv.org/abs/1602.01407) technique to succeed calculations without storing all the Fisher matrix in memory.

**Important :** on the example notebook, some math explanation (to see the simplification in the paper) is available so you can go check that !

## Table of contents

  - [Table of contents](#table-of-contents)
  - [Setup](#setup)
  - [Commands](#commands)
  - [License](#license)

## Setup
You need to have at least Python 3.8 installed on your machine to run the code.

### Without environment
You can install the required packages with this command :
```sh
$ pip3 install numpy scipy, torch, nngeometry # at least
$ pip3 install torchvision # previous command + this one to run the MNIST example
$ pip3 install jupyterlab # previous commands + this one to run MNIST example with Jupyter Notebook
```
**Note :** for Windows systems, change `pip3` to `pip`.

### With environment
You can simply install the packages with this command (make sure that `pipenv` is already installed) :
```sh
$ pipenv install
$ pipenv shell
$ pip install torchvision # previous commands + this one to run the MNIST example
$ pip install jupyterlab # previous commands + this one to run MNIST example with Jupyter Notebook
```

## Commands
Make sure that your environment is activated if you choose to use it.

- To run the script example :
```sh
$ python example/scripts/mnist.py
```

- To run the notebook example :
```sh
$ jupyter lab
```
And go `notebooks/effective_dimension.ipynb` or you can run it via [Google Colab](https://colab.research.google.com/github/Guigui14460/effective-dimension-pytorch/blob/master/examples/notebooks/effective_dimension.ipynb).

## License
Project under the Apache-2.0 license (same as [amyami187/effective_dimension](https://github.com/amyami187/effective_dimension) and [amyami187/local_effective_dimension](https://github.com/amyami187/local_effective_dimension) repositories).
