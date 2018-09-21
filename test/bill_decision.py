import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv("~/Projects/ML/SupLearn/bill_authentication.csv")

#print dataset.shape
#print dataset.head()

X = dataset.drop('Class', axis=1)
y = dataset['Class']

training_accuracy = []
validation_accuracy = []
test_accuracy = []
training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for s in training_size:
    # Define the classifier
    clf = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    clf.fit(X_train, y_train)

    print 'Size: ', s, '%'

    training_accuracy.append(accuracy_score(y_train, clf.predict(X_train)))
    cv = cross_val_score(clf, X_train, y_train, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(y_test, clf.predict(X_test)))

#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

fig = plt.figure()
line2, = plt.plot(training_size, validation_accuracy, 'b', label="Cross Validation Score")
line1, = plt.plot(training_size, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Training size versus Accuracy (Nursery)')
plt.legend(loc='best')
fig.savefig('bill_decision_trainingSize.png')
plt.close(fig)
