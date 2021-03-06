# 10-601 Fall 2014 Assignment 3: Neural Networks #

Primary TAs for this assignment:

- Harry Gifford, hgifford@andrew.cmu.edu
- Jin Sun, jins@andrew.cmu.edu

Please use Piazza if you have any questions.

Please see handout and individual files for more details.

## Directory structure

**This** indicates you should modify this file.
_This_ indicates you may need to modify this file.

Folder                     | Description
------                     | -----------
**./costNN.m**             | Neural network (NN) cost function.
**./computeActivations.m** | Compute the activations of the neural network on the output layer.
**./predictNN.m**          | Use your trained model to return a vector of predicted labels for the given data.
_./trainNN.m_              | Runs the NN training algorithm. You shouldn't need to touch this, unless you are having issues with minFunc. You should still read this function and understand what it is doing though.
./runNN.m                  | Tests your NN on some dataset. You should have first implemented costNN.m, computeActivations.m and predictNN.m
./runDigits.m              | Tests your NN on MNIST. You should be able to run runNN.m before running this.

## Sources:

[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.

[2] An Analysis of Single-Layer Networks in Unsupervised Feature Learning, Adam Coates, Honglak Lee, and Andrew Y. Ng.

[3] Hinton, G. E. & Salakhutdinov, R. R. (2006), 'Reducing the dimensionality of data with neural networks', Science 313 (5786) , 504-507 .