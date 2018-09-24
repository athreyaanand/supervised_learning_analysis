import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

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
layer_values = range(13)

for layer in layer_values:

    hiddens = tuple(layer * [32])
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens, random_state=1)
    clf.fit(train_x, train_y)

    print "layer: ", layer

    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

fig = plt.figure()
plt.style.use('Solarize_Light2')
plt.plot(layer_values, training_accuracy, 'r', label="Training Accuracy")
plt.plot(layer_values, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Hidden Layer Number')
plt.ylabel('Accuracy')
plt.title('Number of Hidden Layer\'s versus Accuracy (Fashion) (32 neurons)')
plt.legend(loc='best')
fig.savefig('images/nn_hidden.png')
plt.close(fig)

'''
training_accuracy = []
validation_accuracy = []
test_accuracy = []
neurons = range(1,33)

for neuron in neurons:
    # Define the classifier
    hiddens = tuple(0 * [neuron])
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens, random_state=1)
    clf.fit(train_x, train_y)

    print 'neuron:', neuron

    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

fig = plt.figure()
plt.style.use('Solarize_Light2')
plt.plot(neurons, training_accuracy, 'r', label="Training Accuracy")
plt.plot(neurons, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.title('Number of Neurons\'s versus Accuracy')
plt.legend(loc='best')
fig.savefig('images/nn_neuron.png')
plt.close(fig)
'''

training_accuracy = []
test_accuracy = []
training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print "***Neural Network***"
for size in training_size:
    hiddens = tuple(0 * [16])
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens, random_state=1)
    t_train, trsh = train_test_split(train, test_size = 1-size)

    t_train_x = t_train.drop('label', axis = 1)
    t_train_y = t_train[label]

    clf.fit(t_train_x, t_train_y)

    print 'Training Size: ', size*100, '%'

    training_accuracy.append(accuracy_score(t_train_y, clf.predict(t_train_x)))
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

fig = plt.figure()
plt.style.use('Solarize_Light2')
plt.plot(training_size, training_accuracy, 'r', label="Training Accuracy")
plt.plot(training_size, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Training size versus Accuracy (Fashion) (0 Hidden)')
plt.legend(loc='best')
fig.savefig('images/nn_trainingSize.png')
plt.close(fig)
