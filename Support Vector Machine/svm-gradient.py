import numpy as np
import math


# Gradient descent for solving support vector regression
def SVR(X_train, Y_train, C, eps):
    w = np.zeros(X_train[0].shape)
    b = 0
    max_pass = 1000
    for t in range(0, max_pass):
        for i in range(len(X_train)):
            eta = 1 / math.sqrt(t + 1)  # decaying step size
            intermediate = Y_train[i] - np.matmul(np.transpose(w), X_train[i]) - b
            if abs(intermediate) >= eps:
                sign = -1 if intermediate >= eps else 1
                w_gradient = sign * X_train[i]
                b_gradient = sign
                w = w - eta * w_gradient
                b = b - eta * b_gradient

            w = 1/(eta + 1) * w
    return w, b

def compute_loss(X, Y, w, b, C, eps):
    regularized = 0.5 * math.pow(np.linalg.norm(w), 2)
    return compute_error(X, Y, w, b, C, eps) + regularized

def compute_error(X, Y, w, b, C, eps):
    sum = 0
    for i in range(len(X)):
        sum += max(0, abs(Y[i] - np.matmul(np.transpose(w), X[i]) - b ) - eps)
    return C * sum

def compute_test_error(pred, Y, w, b, C, eps):
    sum = 0
    for i in range(len(pred)):
        sum += max(0, abs(Y[i] - pred[i] - b ) - eps)

    return C * sum

# Import data
X_train = np.loadtxt('data/X_train_C.csv', delimiter=",")
Y_train = np.loadtxt('data/Y_train_C.csv', delimiter=",")
X_test = np.loadtxt('data/X_test_C.csv', delimiter=",")
Y_test = np.loadtxt('data/Y_test_C.csv', delimiter=",")
C = 1
eps = 0.5

# Run it on Dataset C
w, b = SVR(X_train, Y_train, C, eps)

# report training error, training loss
print(compute_error(X_train, Y_train, w, b, C, eps))
print(compute_loss(X_train, Y_train, w, b, C, eps))

# Run parameters on test data
pred = np.array([np.matmul(test, w) + b for test in X_test])

# report test error using Mean Squared Error
print(compute_test_error(pred, Y_test, w, b, C, eps))