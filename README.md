# Machine Learning Projects

The repository contains projects that utilize various modeling and algorithmic techniques for machine learning since Fall 2023.

# Python packages used

    - Pytorch
    - scikit-learn
    - sklearn
    - statsmodels
    - statistics

# Projects Summary

## Regression

ridge-regression.py: The two implementations of ridge regression (closed-form and gradient descent) and the performance evaluation with the Boston housing datasets.

regression-compare.py: Training of (unregularized) linear regression, ridge regression, and lasso on the datasets A, B, and C along with performance evaluation.

nearest-neighbour.py: The Implementation k-nearest neighbour regression and comparison with linear regression of the performance on different datasets.

## Support Vector Machine

svm-compare.py: Performance comparison between logistic regression, soft SVM, and hard SVM on different datasets.

svm-gradient: The implementation of gradient algorithm for solving a support vector regression and performance analysis.

## Decision Trees

binary-class.py: The implementation of decision tree with different loss function (Gini coefficient, Misclassification, and Entropy) and performance analysis on each approach.

bagging.py: The use of Bagging technique to increase the accuracy of the decision tree using the entropy loss, thus creating an ensemble of 101 decision trees with the maximum depth at 3. 

random-forest.py: The use of random forests method to increase the accuracy of the decision tree, which considers only a random certain number of features when deciding which dimension to split on depend on the dataset.

## Convolutional Neural Network

cnn-vgg11.py: The implementation of VGG11 deep neural network architecture from scratch. The project also trains the model on MNIST dataset with different number of epoch. Trained models are saved as .pth files.

