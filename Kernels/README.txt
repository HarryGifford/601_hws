10-601 Fall 2014 Final Project starter code.
- Harry Gifford (hgifford@cmu.edu)

Please use Piazza if you have any questions.

Please see handout and Kaggle for more details.

This is a small classifier which should give you a rough idea of what to do. We treat each pixel and each color as a dimension. Since images are 32x32 color, that gives us 32*32*3 = 3072 dimensions. This may or may not be a good format in which to classify your data. Look in the handout for tips on how to improve performance.

You are welcome to use as much or as little of this starter code as you wish. The code is here to give you some idea of what to do.

== Directory structure:

Remember, you can type ‘help function_name’ from Matlab/Octave to get help with a function.

./helpers/ - some simple helper functions you might find helpful.

    isOctave.m - returns true if you are running in octave and false if matlab.
    showImages.m - displays a sequence of images of the same size. Useful to visualize the dataset or hidden units in a neural network.
    writeLabels.m - writes out a vector of labels into the format Kaggle expects.

./minFunc/ - optimization function, similar to fminunc or gradient descent, that has excellent performance. To compile type 'addpath ./minFunc' and then type 'mexAll'.

./tinyclassifier/ - simple regularized SVM/LR classifier with support for kernels.

./runClassifier.m - trains and evaluates a the tinyclassifier. You can call it in several ways:

     - runClassifier() will generate some sample data and plot the decision boundary.
     - runClassifier(‘random’) will do the same as above.
     - runClassifier(‘./path/to/data.mat’) will run the classifier on the training and test data in data.mat. data.mat should have X_train, X_test, y_train and optionally, y_test.
     - runClassifier(…, opt) will run in the same way as the above, but you can pass in different parameters through the opt struct. Type ‘help runClassifier’ for a list of what you can change in ‘opt’. You can also manually modify options inside runClassifier.m.
