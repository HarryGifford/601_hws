# 10-601 Fall 2014 Final Project starter code #

- Harry Gifford, hgifford@cmu.edu

Please use Piazza if you have any questions.

Please read the handout and Kaggle for more details.

This is a small classifier which should give you a rough idea of what to do. We treat each pixel and each color as a dimension. Since images are 32x32 color, that gives us 32*32*3 = 3072 dimensions. This may or may not be a good format in which to classify your data. Look in the handout for tips on how to improve performance.

You are welcome to use as much or as little of this starter code as you wish. The code is here to give you some idea of what to do.

## Directory structure:

Remember, you can type 'help function_name' from Matlab/Octave to get help with a function.

Folder                | Description
------                | -----------
./polyKernel.m        | Polynomial Kernel.
./predictClassifier.m | Runs the trained classifier on a particular dataset and returns the predicted labels.
./rbfKernel.m         | RBF kernel.
./runClassifier.m     | Trains and evaluates a the tinyclassifier. You can call it in several ways. 1) runClassifier() will generate some sample data and plot the decision boundary. 2) runClassifier('random') will do the same as above. 3) runClassifier('./path/to/data.mat') will run the classifier on the training and test data in data.mat. data.mat should have X_train, X_test, y_train and optionally, y_test. 4) runClassifier(..., opt) will run in the same way as the above, but you can pass in different parameters through the opt struct. Type 'help runClassifier' for a list of what you can change in 'opt'. You can also manually modify options inside runClassifier.m.
