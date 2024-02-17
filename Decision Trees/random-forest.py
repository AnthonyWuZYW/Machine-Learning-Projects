import numpy as np
import math
import random
import statistics


# Loss function
def Entropy(p):
    if p == 1 or p == 0: return 0
    return -1 * p * math.log2(p) - (1 - p) * math.log2(1 - p)

# modify DecisionTree only consider 4 features when spliting
class DecisionTree_Forest:
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
        # consider only 4 features
        features = random.sample(range(len(X[0])), 4)
        for point_index in range(len(X)):
            # consider only 4 features
            for feature_index in features:
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

        self.left = DecisionTree_Forest(X_left_split, y_left_split, self.depth - 1, self.type)
        self.right = DecisionTree_Forest(X_right_split, y_right_split, self.depth - 1, self.type)

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

# Training with random forest
accuracys = []
for t in range(11):
    bagging_X = []
    bagging_Y = []
    for i in range(101):
        x = []
        y = []
        for j in X_train:
            index = random.randint(0, len(X_train)-1)
            x.append(X_train[index])
            y.append(y_train[index])
        bagging_X.append(x)
        bagging_Y.append(y)

    bagging_trees = []
    for i in range(len(bagging_X)):
        bagging_trees.append(DecisionTree_Forest(bagging_X[i], bagging_Y[i], 3, Entropy))

    pred = []
    for x in X_test:
        predictions = []
        for tree in bagging_trees:
            predictions.append(tree.predict(x))
        pred.append(max(set(predictions), key=predictions.count))

    accuracys.append(accuracy(y_test, pred))

# takes a few miniutes
print("%.2f" % (min(accuracys) * 100), "%")
print("%.2f" % (max(accuracys) * 100), "%")
print("%.2f" % (statistics.median(accuracys) * 100), "%")