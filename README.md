# Machine Learning Projects

The repository contains projects that utilize various modeling and algorithmic techniques for machine learning since Fall 2023.

# Python packages used

    - Pytorch
    - scikit-learn
    - sklearn
    - statsmodels
    - statistics

# Projects Summaries

## Convolutional Neural Network

cnn-vgg11.py: The implementation of VGG11 deep neural network architecture from scratch. The project also trains the model on the MNIST dataset with different epochs. Trained models are saved as .pth files.

cnn-compare.py: Performance comparison between trained VGG11 model with different epochs (loss and accuracy).

cnn-augmentation.py: Performance analysis of trained VGG11 model on augmented data (horizontal flip, vertical flip, blur in different degrees).

## Generative Model

vae.py: The implementation of Variational Autoencoder and generation of sample images.

gan.py: The implementation of Generative Adversarial Network and generation of sample images.

## Adversarial Learning

neural-network.py: A simple neural network architecture with three layers and Relu as the activation function. Trained on MNIST datasets and saved trained model as .pth files.

FGSM.py: The implementation of Fast Gradient Sign Method (FGSM) to generate adversarial images and its effect on performance based on different perturbation budget.

PDG.py: The implementation of Projected Gradient Descent (PDG) to generate adversarial images and training of simple neural network architecture with adversarial images.

## Regression

ridge-regression.py: The two implementations of ridge regression (closed-form and gradient descent) and the performance evaluation with the Boston housing datasets.

regression-compare.py: Training of (unregularized) linear regression, ridge regression, and lasso on the datasets along with performance evaluation.

nearest-neighbour.py: The implementation of k-nearest neighbour regression and comparison with linear regression of the performance on different datasets.

## Support Vector Machine

svm-compare.py: Performance comparison between logistic regression, soft SVM, and hard SVM on different datasets.

svm-gradient: The implementation of gradient algorithm for solving a support vector regression and performance analysis.

## Decision Trees

binary-class.py: The implementation of the Decision Tree with different loss functions (Gini coefficient, Misclassification, and Entropy) and performance analysis on each approach.

bagging.py: The use of the Bagging technique to increase the accuracy of the decision tree using the entropy loss, thus creating an ensemble of 101 decision trees with the maximum depth at 3. 

random-forest.py: The use of the Random Forests method to increase the accuracy of the decision tree, which considers only a random certain number of features when deciding which dimension to split on depending on the dataset.

## Guassian Mixture Model

gmm.py: The implementation of Guassian Mixture Model and the training on the sample dataset. 

