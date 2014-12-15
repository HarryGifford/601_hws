10-601 Fall 2014 Assignment 3: Logistic Regression and Neural Networks.

Primary TAs for this assignment:

Harry Gifford - hgifford@andrew.cmu.edu
Jin Sun       - jins@andrew.cmu.edu

Please use Piazza if you have any questions.

Please see handout and individual files for more details.

— Directory structure:

(*) indicates you should modify this file.
(-) indicates you may need to modify this file.

./helpers          - some simple helper functions used throughout the assignment. No need to worry too much about these.

(*)./costNN.m - neural network (NN) cost function.

(*)./nnComputeActivations.m - compute the activations of the neural network on the output layer and middle layer. Loosely, the activations (a3) from the last layer correspond to unnormalized probabilities in classification and the learned function value in regression.

(*)./nnPredictClassification.m - Use your trained model to return a vector of predicted labels for the given data.

(-)./nnTrain.m - Runs the NN training algorithm. You shouldn’t need to touch this, unless you are having issues with minFunc. You should still read this function and understand what it is doing though.

./nnTrainClassification.m - Wrapper around nnTrain to get a vector of training labels in the correct format for nnTrain.


./runNN.m - Tests your NN on the Banana Dataset. You should have first implemented J.m and nnComputeActivations.m

./runDigits.m           - Tests your NN on MNIST. You should have implemented nnPredictClassification.m before running this.

./runNNRegression.m     - Tests your NN on a simple Regression problem. You should have implemented support for regression in J.m.

- Sources:

[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.

[2] An Analysis of Single-Layer Networks in Unsupervised Feature Learning, Adam Coates, Honglak Lee, and Andrew Y. Ng.

[3] Hinton, G. E. & Salakhutdinov, R. R. (2006), 'Reducing the dimensionality of data with neural networks', Science 313 (5786) , 504-507 .