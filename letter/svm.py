import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_val_score
from numpy import arange

dataset = pd.read_csv("letter.csv")

labelencoder=LabelEncoder()
dataset['lettr'] = labelencoder.fit_transform(dataset['lettr'])

train, test = train_test_split(dataset, test_size = 0.20, random_state = 1)

label = 'lettr'
train_x = train.drop('lettr', axis = 1)
train_y = train[label]
test_x = test.drop('lettr', axis = 1)
test_y = test[label]

training_accuracy = []
validation_accuracy = []
test_accuracy = []
kernels = ['linear','poly', 'rbf']

print "*** SVM ***"
for kernel in kernels:
    if kernel == 'poly':
        clf = svm.SVC(kernel=kernel, degree=2, random_state=1)
    else:
        clf = svm.SVC(kernel=kernel, random_state=1)

    clf.fit(train_x, train_y)
    print kernel

    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    cv = cross_val_score(clf, train_x, train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))


kernels = ['linear','poly (deg:3)', 'rbf']
temp_x = arange(3)
fig = plt.figure()
plt.style.use('Solarize_Light2')
plt.bar(temp_x - 0.3, training_accuracy, width=0.16, color='r', label="Training Accuracy")
plt.bar(temp_x - 0.1, validation_accuracy, width=0.16, color='b', label="Cross Validation Score")
plt.bar(temp_x + 0.1, test_accuracy, width=0.16, color='g', label="Testing Accuracy")
plt.xticks(temp_x, kernels)
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Kernel type versus Accuracy (Letter)')
fig.savefig('images/svm_kernel.png')
plt.close(fig)

training_accuracy = []
validation_accuracy = []
test_accuracy = []
training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for size in training_size:
    clf = svm.SVC(kernel='rbf', random_state=1)
    t_train, trsh = train_test_split(train, test_size = 1-size)

    t_train_x = t_train.drop('lettr', axis = 1)
    t_train_y = t_train[label]

    clf.fit(t_train_x, t_train_y)

    print 'Training Size: ', size*100, '%'

    training_accuracy.append(accuracy_score(t_train_y, clf.predict(t_train_x)))
    cv = cross_val_score(clf, t_train_x, t_train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

fig = plt.figure()
line1, = plt.plot(training_size, training_accuracy, 'r', label="Training Accuracy")
line2, = plt.plot(training_size, validation_accuracy, 'b', label="Cross Validation Score")
line1, = plt.plot(training_size, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Training size versus Accuracy (Letter)')
plt.legend(loc='best')
fig.savefig('images/svm_trainingSize.png')
plt.close(fig)
