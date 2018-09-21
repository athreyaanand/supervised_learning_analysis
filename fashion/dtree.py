import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree._tree import TREE_LEAF

def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)

dataset = pd.read_csv("fashion.csv")

train, test = train_test_split(dataset, test_size = 0.20, random_state = 1)

label = 'label'
train_x = train.drop('label', axis = 1)
train_y = train[label]
test_x = test.drop('label', axis = 1)
test_y = test[label]

training_accuracy = []
validation_accuracy = []
test_accuracy = []
training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for size in training_size:
    clf = DecisionTreeClassifier(max_features='auto', random_state=1)
    t_train, trsh = train_test_split(train, test_size = 1-size)

    t_train_x = t_train.drop('label', axis = 1)
    t_train_y = t_train[label]

    clf.fit(t_train_x, t_train_y)

    print 'Training Size: ', size*100, '%'

    training_accuracy.append(accuracy_score(t_train_y, clf.predict(t_train_x)))
    cv = cross_val_score(clf, t_train_x, t_train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

fig = plt.figure()
plt.style.use('Solarize_Light2')
line2, = plt.plot(training_size, validation_accuracy, 'b', label="Cross Validation Score")
line1, = plt.plot(training_size, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Training size versus Accuracy (Fashions)')
plt.legend(loc='best')
fig.savefig('images/dtree_trainingSize.png')
plt.show()
plt.close(fig)

print "PRUNING"
print(sum(clf.tree_.children_left < 0))
prune_index(clf.tree_, 0, 5)
print(sum(clf.tree_.children_left < 0))
