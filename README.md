# 10-601 Homework Repository #

Contains code for SVM, Logistic Regression and Neural Networks. Written for the Fall 2014 instance of 10-601 Machine Learning at CMU.

## Directory structure

Folder  | Description
------  | -----------
AE      | Autoencoder code. Somewhat messy and probably not useful for use in a homework.
Kernels | Contains code to run Kernelized versions of LR or SVM. Both are trained using gradient methods (either SGD or L-BFGS).
LR      | Logistic Regression code. Can be used to train and test a LR classifier (or SVM).
NN      | Neural Network code.
data    | Contains some datasets used by the different classifiers.
shared  | Some helper functions used all over the code.

## Getting Started

### Setup `minFunc`.
This tool is used for solving the optimization problems. Open Matlab or Octave in the root directory, then type the following:
```matlab
cd ./shared;
addpath ./minFunc;
mexAll ./minFunc;
rmpath ./minFunc;
cd ..
```
The output should look like:
```
Compiling minFunc files (octave version)...
mcholC compiled
lbfgsC compiled
lbfgsAddC compiled
lbfgsProdC compiled
Done.
```
### Train and test sample data
Most of the directories contain a function `runX`, such as `runLR` or `runNN` which will generate some data, train the model, and display the decision boundary. Look at the README in Kernels, for some detailed examples.
### Train and test MNIST
Most of the directories contain a function `runDigits`, which will train and test the relevant classifer on the MNIST dataset and display the misclassified digits.
