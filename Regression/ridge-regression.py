import numpy as np
import math
import helper
import matplotlib.pyplot as plt
import time

# Implement ridge regression
def ridge_regression(X, Y, lamb):
    n = len(X)
    one = np.ones((X.shape[0], 1))
    zero = np.zeros((X.shape[1], 1))
    I = np.identity(X.shape[1])

    A1 = np.concatenate((X, one), axis=1)
    A2 = np.concatenate((math.sqrt(2 * lamb * n) * I, zero), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    AT = np.transpose(A)

    z = np.concatenate((Y, zero), axis=0)

    solution = np.linalg.solve(np.matmul(AT, A), np.matmul(AT, z))

    return solution[:-1], solution[-1]

# Implement gradient descent for ridge regression
def gradient_descent(X, Y, w, b, lamb, max, eta, tol):
    n = len(X)
    one = np.ones((X.shape[0], 1))
    zero = np.zeros((X.shape[1], 1))
    I = np.identity(X.shape[1])
    train_loss = []

    A1 = np.concatenate((X, one), axis=1)
    A2 = np.concatenate((math.sqrt(2 * lamb * n) * I, zero), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    AT = np.transpose(A)
    z = np.concatenate((Y, zero), axis=0)

    wt = np.concatenate((w, np.array([[b]])), axis=0)
    for i in range(max):
        loss = (np.matmul(np.matmul(AT, A), wt) - np.matmul(AT, z))/n
        wt1 = wt - eta * loss
        train_loss.append(training_loss(X, Y, wt1[:-1], wt1[-1], lamb))
        if np.linalg.norm(wt - wt1) <= tol:
            break
        wt = wt1
    solution = wt1
    return solution[:-1], solution[-1], train_loss

# standardize the data features
def standardize(X):
    XT = np.transpose(X)
    mean = [sum(XT[i])/float(len(XT[i])) for i in range(XT.shape[0])]
    stdev = [np.sqrt(sum(np.power(XT[i] - mean[i], 2))/float(len(XT[i]))) for i in range(XT.shape[0])]
    for row in X:
        for i in range(len(row)):
            row[i] = (row[i] - mean[i]) / stdev[i]
    return X


# Test two implementations on the Boston housing dataset

# Calculatie training_error
def training_error(X, Y, w, b):
    n = len(X)
    one = np.ones((X.shape[0], 1))
    error = np.matmul(X, w) + one * b - Y
    return np.power(np.linalg.norm(error), 2)/ (2 * n)


# Calculate training_loss
def training_loss(X, Y, w, b, lamb):
    return training_error(X, Y, w, b) + lamb * np.power(np.linalg.norm(w), 2)

# Caculate Mean Squared Error
def mean_squared_error(Y, predict):
    diff = np.power(np.ndarray.flatten(Y)-np.ndarray.flatten(predict), 2)
    return sum(diff)/len(diff)

#load in data
training_x_file = "housing_X_train.csv"
training_y_file = "housing_y_train.csv"
testing_x_file = "housing_X_test.csv"
testing_y_file = "housing_y_test.csv"

trainX, trainY = helper.load_data(training_x_file, training_y_file)
testX, testY = helper.load_data(testing_x_file, testing_y_file)

# standardize the data for gradient descent
trainX_std = standardize(trainX)
testX_std = standardize(testX)

# initalize w, b
wi = np.zeros((trainX.shape[1], 1))
bi = 0

#step size,tol, max_step
eta = 0.01
tol = 0.0001
max_step = 1000

# lambda = 0
l1 = 0

start = time.time()
w1, b1 = ridge_regression(trainX, trainY, l1)
end1 = time.time()
w2, b2, loss2 = gradient_descent(trainX_std, trainY, wi, bi, l1, max_step, eta, tol)
end2 = time.time()

time1 = end1 - start
time2 = end2 - end1

# predicted y by solving closed form solution for ridge regression with lambda = 10
predict1 = np.array([np.matmul(test, w1) + b1 for test in testX])
# predicted y by using gradient_descent to solve ridge regression with lambda = 10
predict2 = np.array([np.matmul(test, w2) + b2 for test in testX_std])

# predict the median house price
median1 = np.median(predict1)
median2 = np.median(predict2)
print(median1)
print(median2)

# report training error, training loss and test error(mean squared error)
train_error1 = training_error(trainX, trainY, w1, b1)
train_loss1 = training_loss(trainX, trainY, w1, b1, l1)
test_error1 = mean_squared_error(testY, predict1)
print(train_error1, train_loss1, test_error1)

train_error2 = training_error(trainX, trainY, w2, b2)
train_loss2 = training_loss(trainX, trainY, w2, b2, l1)
test_error2 = mean_squared_error(testY, predict2)
print(train_error2, train_loss2, test_error2)

# lambda = 10
l2 = 10

start2 = time.time()
w3, b3 = ridge_regression(trainX, trainY, l2)
end3 = time.time()
w4, b4, loss4 = gradient_descent(trainX_std, trainY, wi, bi, l2, max_step, eta, tol)
end4 = time.time()

time3 = end3 - start2
time4 = end4 - end3

# predicted y by solving closed form solution for ridge regression with lambda = 10
predict3 = np.array([np.matmul(test, w3) + b3 for test in testX])
# predicted y by using gradient_descent to solve ridge regression with lambda = 10
predict4 = np.array([np.matmul(test, w4) + b4 for test in testX_std])

# predict the median house price
median3 = np.median(predict3)
median4 = np.median(predict4)
print(median3)
print(median4)

# report training error, training loss and test error(mean squared error)
train_error3 = training_error(trainX, trainY, w3, b3)
train_loss3 = training_loss(trainX, trainY, w3, b3, l2)
test_error3 = mean_squared_error(testY, predict3)
print(train_error3, train_loss3, test_error3)

train_error4 = training_error(trainX, trainY, w4, b4)
train_loss4 = training_loss(trainX, trainY, w4, b4, l2)
test_error4 = mean_squared_error(testY, predict4)
print(train_error4, train_loss4, test_error4)


fig, axis = plt.subplots(2)
fig.set_figheight(8)
axis[0].plot(np.array([i for i in range(len(loss2))]), np.array(loss2))        #show plot when lambda = 0
axis[1].plot(np.array([i for i in range(len(loss4))]), np.array(loss4))        #show plot when lambda = 10
axis[0].set_xlabel("iterations")
axis[0].set_ylabel("training loss")
axis[1].set_xlabel("iterations")
axis[1].set_ylabel("training loss")
plt.show()


print("Closed form:" + str(time1), "Gradient Descent:" + str(time2))
print("Closed form:" + str(time3), "Gradient Descent:" + str(time4))
