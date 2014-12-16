# 10-601 Fall 2014 Assignment 3: Logistic Regression #

- Harry Gifford (hgifford@cmu.edu)

Please use Piazza if you have any questions.

## Directory structure

**This** indicates you should modify this file.
_This_ indicates you may need to modify this file.

Folder          | Description
------          | -----------
costL2.m        | L2 Regularization cost function. Used to add regularization to a cost function.
**costLR.m**    | Logistic Regression cost function.
costSE.m        | Squared error cost function.
costSVM.m       | SVM cost function.
**minimize.m**  | Function applies gradient descent to the given cost function.
**predictLR.m** | Predicts class labels for data given a trained model.
runDigits.m     | Script that trains and tests your logistic regression classifier against a subset of the MNIST dataset.
runLR.m         | Script that trains and tests your logistic regression classifier against some dataset. If no dataset is given it runs against a randomly generated 2D dataset and plots the decision boundary.
trainLR.m       | Trains the Logistic Regression classifier. You can play with the code and drop in any of costLR, costSE or costSVM to change the classifier.
