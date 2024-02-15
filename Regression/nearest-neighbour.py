import helper
import random
import math
import numpy as np
import matplotlib.pyplot as plt


# Calculating the l2 distance as the distance metric
def dist(X1, X2):
    distance = 0
    for i in range(len(X1)):
        distance += math.pow(float(X1[i]) - float(X2[i]), 2)
    return math.sqrt(distance)

# Quick select Implementation
def partition(lst, left, right):
    index = random.randint(left, right)
    pivot = lst[index]
    lst[right], lst[index] = lst[index], lst[right]
    i = left - 1
    for j in range(left, right):
        if lst[j] < pivot:
            i = i + 1
            lst[i], lst[j] = lst[j], lst[i]
    lst[i+1], lst[right] = lst[right], lst[i+1]
    return i + 1

def quick_select(lst, k, left, right):
    if left == right:
        return lst[left]
    p = partition(lst, left, right)
    i = p - left + 1
    if k == i:
        return lst[p]
    if k < i:
        return quick_select(lst, k, left, p - 1)
    else:
        return quick_select(lst, k - i, p + 1, right)

# Implementation of k-nearest neighbour classification
def kNN(Xtrain, Ytrain, x, k):
    length = len(Ytrain)
    distances = []
    neighbours =[]
    for i in range(length):
        distance = dist(x, Xtrain[i])
        distances.append(distance)

    kth_smallest = quick_select(distances.copy(), k, 0, len(distances)-1)

    k_distance = []
    for i in range(length):
        if distances[i] <= kth_smallest:
            k_distance.append(distances[i])
            neighbours.append(float(Ytrain[i][0]))

    # use to inspecting distances between the test points and their k nearest neighbour
    return sum(neighbours)/len(neighbours)


# linear regression
def linear_regression(X, Y):
    n = len(X)
    one = np.ones((X.shape[0], 1))
    zero = np.zeros((X.shape[1], 1))
    I = np.identity(X.shape[1])

    A1 = np.concatenate((X, one), axis=1)
    A2 = np.concatenate((0 * I, zero), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    AT = np.transpose(A)

    z = np.concatenate((Y, zero), axis=0)

    solution = np.linalg.solve(np.matmul(AT, A), np.matmul(AT, z))

    return solution[:-1], solution[-1]

# mean-squared error
def mean_squared_error(Y, predict):
    diff = np.power(np.ndarray.flatten(Y)-np.ndarray.flatten(predict), 2)
    return sum(diff)/len(diff)

D_X_train, D_Y_train = helper.load_data("data/X_train_D.csv", "data/Y_train_D.csv")
E_X_train, E_Y_train = helper.load_data("data/X_train_E.csv", "data/Y_train_E.csv")

D_X_test, D_Y_test = helper.load_data("data/X_test_D.csv", "data/Y_test_D.csv")
E_X_test, E_Y_test = helper.load_data("data/X_test_E.csv", "data/Y_test_E.csv")


# Compare with linear regression solution
w_D, b_D = linear_regression(D_X_train, D_Y_train)
linear_predict_D = np.array([np.matmul(test, w_D) + b_D for test in D_X_test])
# Calculate error
linear_error_D = mean_squared_error(D_Y_test, linear_predict_D)

w_E, b_E = linear_regression(E_X_train, E_Y_train)
linear_predict_E = np.array([np.matmul(test, w_E) + b_E for test in E_X_test])
# Calculate error
linear_error_E = mean_squared_error(E_Y_test, linear_predict_E)

knn_pred_D = []
test_error_D =[]
knn_pred_E = []
test_error_E =[]

# knn_pred of size 9 has the k-nearest neighbour regression solutions
# with each integer k from 1 to 9.
for k in range(1,10):
    knn_pred_D.append([kNN(D_X_train, D_Y_train, x, k) for x in D_X_test])
    test_error_D.append(mean_squared_error(D_Y_test, np.array(knn_pred_D[k-1])))
    knn_pred_E.append([kNN(E_X_train, E_Y_train, x, k) for x in E_X_test])
    test_error_E.append(mean_squared_error(E_Y_test, np.array(knn_pred_E[k - 1])))

fig, axis = plt.subplots(2,2)
fig.set_figheight(8)
fig.set_figwidth(12)

axis[0,0].scatter(np.ndarray.flatten(D_X_test), linear_predict_D, facecolors='none', edgecolors="red", label="least squares")
axis[0,0].scatter(np.ndarray.flatten(D_X_test), knn_pred_D[0],  facecolors='none', edgecolors="blue", label="1-nearest neighbour")
axis[0,0].scatter(np.ndarray.flatten(D_X_test), knn_pred_D[8],facecolors='none', edgecolors="pink",  label="9-nearest neighbour")
axis[0,0].legend()
axis[0,0].set_title("Data Set D with different solution")

axis[1,0].plot([k for k in range(1,10)], test_error_D)
axis[1,0].axhline(y = linear_error_D)
axis[1,0].set_title("Data Set D Test Error")

axis[0,1].scatter(np.ndarray.flatten(E_X_test), linear_predict_E, facecolors='none', edgecolors="red", label="least squares")
axis[0,1].scatter(np.ndarray.flatten(E_X_test), knn_pred_E[0],  facecolors='none', edgecolors="blue", label="1-nearest neighbour")
axis[0,1].scatter(np.ndarray.flatten(E_X_test), knn_pred_E[8],facecolors='none', edgecolors="pink",  label="9-nearest neighbour")
axis[0,1].legend()
axis[0,1].set_title("Data Set E with different solution")

axis[1,1].plot([k for k in range(1,10)], test_error_E)
axis[1,1].axhline(y = linear_error_E)
axis[1,1].set_title("Data Set E Test Error")

plt.show()


# Test on datasets with features size of 20
F_X_train, F_Y_train = helper.load_data("data/X_train_F.csv", "data/Y_train_F.csv")

F_X_test, F_Y_test = helper.load_data("data/X_test_F.csv", "data/Y_test_F.csv")
w_F, b_F = linear_regression(F_X_train, F_Y_train)
linear_predict_F = np.array([np.matmul(test, w_F) + b_F for test in F_X_test])
# Calculate error
linear_error_F = mean_squared_error(F_Y_test, linear_predict_F)

knn_pred_F = []
test_error_F =[]

# knn_pred of size 9 has the k-nearest neighbour regression solutions
# with each integer k from 1 to 9.
for k in range(1,10):
    knn_pred_F.append([kNN(F_X_train, F_Y_train, x, k) for x in F_X_test])
    test_error_F.append(mean_squared_error(F_Y_test, np.array(knn_pred_F[k-1])))

# close the first plot to see this one
plt.plot([k for k in range(1,10)], test_error_F)
plt.axhline(y = linear_error_F)
plt.title("Data Set F Test Error")
plt.show()