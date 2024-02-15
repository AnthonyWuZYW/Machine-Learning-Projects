import csv
import numpy as np

def split(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]

def load_data(feature_file, label_file):
    features = []
    with open(feature_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            features += [float(x) for x in row]

    labels = []
    with open(label_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append([float(x) for x in row])

    featureSet = split(features, int(len(features)/len(labels)))


    return np.array(featureSet), np.array(labels)

