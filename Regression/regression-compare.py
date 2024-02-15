import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
import matplotlib.pyplot as plt
import helper

A_X_train, A_Y_train = helper.load_data("X_train_A.csv", "Y_train_A.csv")
B_X_train, B_Y_train = helper.load_data("X_train_B.csv", "Y_train_B.csv")
C_X_train, C_Y_train = helper.load_data("X_train_C.csv", "Y_train_C.csv")

A_X_test, A_Y_test = helper.load_data("X_test_A.csv", "Y_test_A.csv")
B_X_test, B_Y_test = helper.load_data("X_test_B.csv", "Y_test_B.csv")
C_X_test, C_Y_test = helper.load_data("X_test_C.csv", "Y_test_C.csv")


# Caculate Mean Squared Error
def mean_squared_error(Y, predict):
    diff = np.power(np.ndarray.flatten(Y)-np.ndarray.flatten(predict), 2)
    return sum(diff)/len(diff)


# Get get optimal lambda for ridge regression use k-fold cross validation
def ridge_CV(X_train, Y_train, k, lamb_range):
    split_x = np.split(X_train, k)
    split_y = np.split(Y_train, k)
    perfs = []
    for l in lamb_range:
        perf_sum = 0
        for f in range(k):
            start = False
            for i in range(len(split_x)):
                if i != f:
                    if not start:
                        start = True
                        fold_x = split_x[i].copy()
                        fold_y = split_y[i].copy()
                    else:
                        fold_x = np.concatenate((fold_x, split_x[i]))
                        fold_y = np.concatenate((fold_y, split_y[i]))
            # use
            clf = Ridge(alpha=l)
            clf.fit(fold_x, fold_y)
            pred = clf.predict(split_x[f])
            perf_sum += -1 * mean_squared_error(pred, split_y[f])
        perfs.append(perf_sum)

    return lamb_range[np.argmax(perfs)]


# Get get optimal lambda for ridge regression use k-fold cross validation
def lasso_CV(X_train, Y_train, k, lamb_range):
    split_x = np.split(X_train, k)
    split_y = np.split(Y_train, k)
    perfs = []
    for l in lamb_range:
        perf_sum = 0
        for f in range(k):
            start = False
            for i in range(len(split_x)):
                if i != f:
                    if not start:
                        start = True
                        fold_x = split_x[i].copy()
                        fold_y = split_y[i].copy()
                    else:
                        fold_x = np.concatenate((fold_x, split_x[i]))
                        fold_y = np.concatenate((fold_y, split_y[i]))
            # use
            clf = linear_model.Lasso(alpha=l)
            clf.fit(fold_x, fold_y)
            pred = clf.predict(split_x[f])
            perf_sum += -1 * mean_squared_error(pred, split_y[f])
        perfs.append(perf_sum)

    return lamb_range[np.argmax(perfs)]


# For linear regression

regA = LinearRegression().fit(A_X_train, A_Y_train)
predA = regA.predict(A_X_test)
linear_error_A = mean_squared_error(A_Y_test, predA)
print(linear_error_A)

regB = LinearRegression().fit(B_X_train, B_Y_train)
predB = regB.predict(B_X_test)
linear_error_B = mean_squared_error(B_Y_test, predB)
print(linear_error_B)

regC = LinearRegression().fit(C_X_train, C_Y_train)
predC = regC.predict(C_X_test)
linear_error_C = mean_squared_error(C_Y_test, predC)
print(linear_error_C)

avg_linear_mean_squared_error = (linear_error_A + linear_error_B + linear_error_C ) /3
print(avg_linear_mean_squared_error)



# For ridge regression

ridge_lamb_A = ridge_CV(A_X_train, A_Y_train, 5, [8, 9, 10])
ridge_lamb_B = ridge_CV(B_X_train, B_Y_train, 5, [54, 55, 56])
ridge_lamb_C = ridge_CV(C_X_train, C_Y_train, 5, [0.1, 0.5, 1])

ridge_clf_A = linear_model.Ridge(alpha=ridge_lamb_A)
ridge_clf_A.fit(A_X_train, A_Y_train)
r_predA = ridge_clf_A.predict(A_X_test)
ridge_error_A = mean_squared_error(A_Y_test, r_predA)
print(ridge_error_A)

ridge_clf_B = linear_model.Ridge(alpha=ridge_lamb_B)
ridge_clf_B.fit(B_X_train, B_Y_train)
r_predB = ridge_clf_B.predict(B_X_test)
ridge_error_B = mean_squared_error(B_Y_test, r_predB)
print(ridge_error_B)

ridge_clf_C = linear_model.Ridge(alpha=ridge_lamb_C)
ridge_clf_C.fit(C_X_train, C_Y_train)
r_predC = ridge_clf_C.predict(C_X_test)
ridge_error_C = mean_squared_error(C_Y_test, r_predC)
print(ridge_error_C)

avg_ridge_mean_squared_error = (ridge_error_A + ridge_error_B + ridge_error_C ) /3
print(avg_ridge_mean_squared_error)

# For  lasso

lasso_lamb_A = lasso_CV(A_X_train, A_Y_train, 5, [0.001, 0.01, 0.1])
lasso_lamb_B = lasso_CV(B_X_train, B_Y_train, 5, [0.01, 0.05, 0.1])
lasso_lamb_C = lasso_CV(C_X_train, C_Y_train, 5, [0.01, 0.1, 1])

lasso_clf_A = linear_model.Lasso(alpha=lasso_lamb_A)
lasso_clf_A.fit(A_X_train, A_Y_train)
l_predA = lasso_clf_A.predict(A_X_test)
lasso_error_A = mean_squared_error(A_Y_test, l_predA)
print(lasso_error_A)

lasso_clf_B = linear_model.Lasso(alpha=lasso_lamb_B)
lasso_clf_B.fit(B_X_train, B_Y_train)
l_predB = lasso_clf_B.predict(B_X_test)
lasso_error_B = mean_squared_error(B_Y_test, l_predB)
print(lasso_error_B)

lasso_clf_C = linear_model.Lasso(alpha=lasso_lamb_C)
lasso_clf_C.fit(C_X_train, C_Y_train)
l_predC = lasso_clf_C.predict(C_X_test)
lasso_error_C = mean_squared_error(C_Y_test, l_predC)
print(lasso_error_C)

avg_lasso_mean_squared_error = (lasso_error_A + lasso_error_B + lasso_error_C ) /3
print(avg_lasso_mean_squared_error)

# Create histogram

figure, axis = plt.subplots(2, 2)

# For dataset A
axis[0, 0].hist([np.ndarray.flatten(regA.coef_),np.ndarray.flatten(ridge_clf_A.coef_), np.ndarray.flatten(lasso_clf_A.coef_)  ], 20, edgecolor="black", label=["linear", "ridge", "lasso"])
axis[0, 0].set_xlabel("values of coordinates")
axis[0, 0].set_ylabel("number of coordinates")
axis[0, 0].set_title("Data Set A")
axis[0, 0].legend()

# For dataset B
axis[0, 1].hist([np.ndarray.flatten(regB.coef_),np.ndarray.flatten(ridge_clf_B.coef_), np.ndarray.flatten(lasso_clf_B.coef_)  ], 20, edgecolor="black", label=["linear", "ridge", "lasso"])
axis[0, 1].set_xlabel("values of coordinates")
axis[0, 1].set_ylabel("number of coordinates")
axis[0, 1].set_title("Data Set B")
axis[0, 1].legend()

# For dataset C
axis[1, 0].hist([np.ndarray.flatten(regC.coef_),np.ndarray.flatten(ridge_clf_C.coef_), np.ndarray.flatten(lasso_clf_C.coef_)  ], 20, edgecolor="black", label=["linear", "ridge", "lasso"])
axis[1, 0].set_xlabel("values of coordinates")
axis[1, 0].set_ylabel("number of coordinates")
axis[1, 0].set_title("Data Set C")
axis[1, 0].legend()

plt.show()

