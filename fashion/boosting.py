import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

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
n_estim = range(1,51)

for n in n_estim:
    clf = AdaBoostClassifier(n_estimators=n, random_state=1)
    clf.fit(train_x, train_y)
    print "Number of estimators: ", n

    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    cv = cross_val_score(clf, train_x, train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

fig = plt.figure()
plt.style.use('Solarize_Light2')
line1, = plt.plot(n_estim, training_accuracy, 'r', label="Training Accuracy")
line2, = plt.plot(n_estim, validation_accuracy, 'b', label="Cross Validation Score")
line1, = plt.plot(n_estim, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Number of Estimators versus Accuracy (Letter)')
fig.savefig('images/bosting_estimator.png')
plt.close(fig)

training_accuracy = []
validation_accuracy = []
test_accuracy = []
training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for size in training_size:
    clf = AdaBoostClassifier(n_estimators=50, random_state=1) 
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
plt.style.use('Solarize_Light2')
line1, = plt.plot(training_size, training_accuracy, 'r', label="Training Accuracy")
line2, = plt.plot(training_size, validation_accuracy, 'b', label="Cross Validation Score")
line1, = plt.plot(training_size, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Training size versus Accuracy (Letter)')
plt.legend(loc='best')
fig.savefig('images/bosting_trainingSize.png')
plt.close(fig)
