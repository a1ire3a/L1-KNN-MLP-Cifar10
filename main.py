import pickle
import numpy as np
from sklearn import model_selection
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

def unpickle(file):

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

file1 = r'...\cifar-10-batches-py\data_batch_1'
file0 = r'...\cifar-10-batches-py\batches.meta'
data_batch_1 = unpickle(file1)
batches = unpickle(file0)

data_train, data_test, label_train, label_test = model_selection.train_test_split(data_batch_1[b'data'], data_batch_1[b'labels'], test_size=0.3)
data_train = np.asarray(data_train)
data_test = np.asarray(data_test)
label_train = np.asarray(label_train)
label_test = np.asarray(label_test)

len_test = len(data_test)
len_train = len(data_train)
len_sample = len(data_test[0])

data_train_s = preprocessing.MinMaxScaler().fit_transform(data_train)
data_test_s = preprocessing.MinMaxScaler().fit_transform(data_test)

#l1 distance
x = 255*3072
similar = 0
predict_label = -1
for i in range(len_test):
    for j in range(len_train):
        temp = np.asarray(abs(data_test[i] - data_train[j]))
        l1 = np.sum(temp)
        if l1 < x:
            x = l1
            predict_label = label_train[j]
    if predict_label == label_test[i]:
        similar = similar + 1
print(similar / len_test)

#KNN
def predict(x_train, y_train, x_test, k):
    labels = []
    for ii in range(len(x_test)):
        point_dist = []
        for jj in range(len(x_train)):
            distances = np.sqrt(np.sum((x_train[jj]-x_test[ii])**2))
            point_dist.append(distances)
        point_dist = np.array(point_dist)
        dist = np.argsort(point_dist)[:k]
        label = y_train[dist]
        majority_lab = mode(labels)
        majority_lab = majority_lab.mode[0]
        labels.append(np.asarray(majority_lab))
    return labels

knum = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
for i in knum:
    print(i)
    pd_label = np.asarray(predict(data_train_s, label_train, data_test_s, i))
    print(accuracy_score(label_test, pd_label))

#MLP
num = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for i in num:
    clf = MLPClassifier(hidden_layer_sizes=100, max_iter=10000)
    clf.fit(data_train_s, label_train)
    pr = clf.predict(data_test_s)
    acc = accuracy_score(label_test, pr)
    print(acc)
