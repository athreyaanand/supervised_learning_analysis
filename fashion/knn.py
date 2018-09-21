import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv("fashion.csv")

labelencoder=LabelEncoder()
dataset['label'] = labelencoder.fit_transform(dataset['label'])

train, test = train_test_split(dataset, test_size = 0.20, random_state = 1)

label = 'label'
train_x = train.drop('label', axis = 1)
train_y = train[label]
test_x = test.drop('label', axis = 1)
test_y = test[label]

training_accuracy = []
validation_accuracy = []
test_accuracy = []
k_vals = range(1,35,2)

# For knn, experiment on different k values
for k in k_vals:

    # Define the classifier
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(train_x, train_y)

    print 'Neighbors: ', k

    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    cv = cross_val_score(clf, train_x, train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

# Plot the k values graph
fig = plt.figure()
plt.style.use('Solarize_Light2')
line1, = plt.plot(k_vals, training_accuracy, 'r', label="Training Accuracy")
line2, = plt.plot(k_vals, validation_accuracy, 'b', label="Cross Validation Score")
line1, = plt.plot(k_vals, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('K-Nearest')
plt.ylabel('Accuracy')
plt.title('Number of K\'s versus Accuracy (Fashion)')
plt.legend(loc='best')
fig.savefig('images/knn_number.png')
plt.close(fig)

k = 2
training_accuracy = []
validation_accuracy = []
test_accuracy = []
training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print "***KNN***"
for size in training_size:
    clf = neighbors.KNeighborsClassifier(k) 
    t_train, trsh = train_test_split(train, test_size = 1-size)

    t_train_x = t_train.drop('label', axis = 1)
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
plt.title('Training size versus Accuracy (Fashion)')
plt.legend(loc='best')
fig.savefig('images/knn_trainingSize.png')
plt.close(fig)
