import numpy as np
import statsmodels.api as sm
from sklearn import svm
import sys

# Load the files
X_train_A = np.loadtxt('data/X_train_A.csv', delimiter=",")
Y_train_A = np.loadtxt('data/Y_train_A.csv',  delimiter=",").astype(int)

X_train_B = np.loadtxt('data/X_train_B.csv', delimiter=",")
Y_train_B = np.loadtxt('data/Y_train_B.csv', delimiter=",").astype(int)
X_test_B = np.loadtxt('data/X_test_B.csv', delimiter=",")
Y_test_B = np.loadtxt('data/Y_test_B.csv', delimiter=",").astype(int)


# Run logistic regression on Dataset A
# log_reg = sm.Logit(Y_train_A, X_train_A).fit()
# failed due to dataset being "Perfect Separation"

# Run SVM with l2 regularization with parameter 1 on Dataset A
clf1 = svm.SVC(C = 1, kernel="linear")
clf1.fit(X_train_A, Y_train_A)

# Run SVM with regularization parameter float(’inf’) on Dataset A
clf2 = svm.SVC(C=sys.float_info.max, kernel="linear")
clf2.fit(X_train_A, Y_train_A)


# Compute for the same three methods on Dataset B
# Run logistic regression on Dataset B
log_regB = sm.Logit(Y_train_B, X_train_B).fit()

# Run SVM with l2 regularization with parameter 1 on Dataset B
clf3 = svm.SVC(C = 1, kernel="linear")
clf3.fit(X_train_B, Y_train_B)

# Run SVM with regularization parameter float(’inf’) on Dataset B
# clf4 = svm.SVC(C=sys.float_info.max, kernel="linear")
# clf4.fit(X_train_B, Y_train_B)
# failed due to Hard SVM doesn't allow misclassification

# the empirical prediction accuracy
predicted = clf3.predict(X_test_B)
match = 0
for i in range(len(Y_test_B)):
    if Y_test_B[i] == predicted[i]:
        match += 1
print(match/len(Y_test_B)*100, "%")