import numpy as np
import math
import matplotlib.pyplot as plt


# Loss functions
def Misclass(p):
    return min(p, 1 - p)

def Gini(p):
    return p * (1 - p)

def Entropy(p):
    if p == 1 or p == 0: return 0
    return -1 * p * math.log2(p) - (1 - p) * math.log2(1 - p)

class DecisionTree:
    def __init__(self, x_train, y_train, depth, type):
        self.X = x_train
        self.y = y_train
        self.depth = depth
        self.left = None
        self.right = None
        self.split = None
        self.split_index = None
        self.type = type
        self.build(self.X, self.y)

    def build(self, X, y):
        if len(set(y)) <= 1 or self.depth == 0:
            return

        loss = float('inf')
        X_left_split = []
        y_left_split = []
        X_right_split = []
        y_right_split = []
        y_len = len(y)
        for point_index in range(len(X)):
            for feature_index in range(len(X[point_index])):
                j = X[point_index][feature_index]
                X_left = []
                y_left = []
                X_right = []
                y_right = []
                # splitting
                for index in range(len(X)):
                    if X[index][feature_index] <= j:
                        X_left.append(X[index])
                        y_left.append(y[index])
                    else:
                        X_right.append(X[index])
                        y_right.append(y[index])
                # Calculating loss
                len_left = len(y_left)
                len_right = len(y_right)
                if len_right * len_left == 0:
                    continue
                p_left = y_left.count(1) / len_left
                p_right = y_right.count(1) / len_right

                error = len_left / y_len * self.type(p_left) + len_right / y_len * self.type(p_right)

                if error < loss:
                    loss = error
                    self.split = j
                    self.split_index = feature_index
                    X_left_split = X_left
                    y_left_split = y_left
                    X_right_split = X_right
                    y_right_split = y_right

        self.left = DecisionTree(X_left_split, y_left_split, self.depth - 1, self.type)
        self.right = DecisionTree(X_right_split, y_right_split, self.depth - 1, self.type)

    def predict(self, X):
        if not self.left:
            return max(set(self.y), key=list(self.y).count)

        if X[self.split_index] <= self.split:
            return self.left.predict(X)
        else:
            return self.right.predict(X)
        
# Calculate the accuracy of the prediction
def accuracy(y, predict):
    correct = 0
    total = len(y)
    for i in range(total):
        if y[i] == predict[i]:
            correct += 1
    return correct/total

# Load data
X_train = np.loadtxt('data/X_train_D.csv', delimiter=",")
y_train = np.loadtxt('data/y_train_D.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test_D.csv', delimiter=",")
y_test = np.loadtxt('data/y_test_D.csv', delimiter=",").astype(int)

# tree depths
depth = [i for i in range(21)]

# Misclassification accuracy
training_accuracy1 = []
test_accuracy1 = []

# Gini coefficient accuracy
training_accuracy2 = []
test_accuracy2 = []

# Entropy accuracy
training_accuracy3 = []
test_accuracy3 = []

for d in depth:
    # Misclassification
    tree1 = DecisionTree(X_train, y_train, d, Misclass)
    pred_train1 = np.array([tree1.predict(train) for train in X_train])
    pred_test1 = np.array([tree1.predict(test) for test in X_test])

    training_accuracy1.append(accuracy(y_train, pred_train1))
    test_accuracy1.append(accuracy(y_test, pred_test1))

    # Gini coefficient
    tree2 = DecisionTree(X_train, y_train, d, Gini)
    pred_train2 = np.array([tree2.predict(train) for train in X_train])
    pred_test2 = np.array([tree2.predict(test) for test in X_test])

    training_accuracy2.append(accuracy(y_train, pred_train2))
    test_accuracy2.append(accuracy(y_test, pred_test2))

    # Entropy
    tree3 = DecisionTree(X_train, y_train, d, Entropy)
    pred_train3 = np.array([tree3.predict(train) for train in X_train])
    pred_test3 = np.array([tree3.predict(test) for test in X_test])

    training_accuracy3.append(accuracy(y_train, pred_train3))
    test_accuracy3.append(accuracy(y_test, pred_test3))

# plot the accuracies
fig, axis = plt.subplots(2,2)
fig.set_figheight(8)
fig.set_figwidth(12)

axis[0,0].plot(training_accuracy1)
axis[0,0].plot(test_accuracy1)
axis[0,0].set_title("Decision Trees with Misclassification Error as Loss function")
axis[0,0].legend(["Training Accuracy", "Test Accuracy"], loc ="lower right")
axis[0,0].set_xlabel("Maximum Depth")
axis[0,0].set_ylabel("Accuracy")

axis[1,0].plot(training_accuracy2)
axis[1,0].plot(test_accuracy2)
axis[1,0].set_title("Decision Trees with Gini Index as Loss function")
axis[1,0].legend(["Training Accuracy", "Test Accuracy"], loc ="lower right")
axis[1,0].set_xlabel("Maximum Depth")
axis[1,0].set_ylabel("Accuracy")

axis[0,1].plot(training_accuracy3)
axis[0,1].plot(test_accuracy3)
axis[0,1].set_title("Decision Trees with Entropy as Loss function")
axis[0,1].legend(["Training Accuracy", "Test Accuracy"], loc ="lower right")
axis[0,1].set_xlabel("Maximum Depth")
axis[0,1].set_ylabel("Accuracy")

plt.show()

