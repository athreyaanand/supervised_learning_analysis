import matplotlib.pyplot as plt
from numpy import arange

letter = [82.5, 27.5, 77, 97.1, 93.9]
fashion = [72, 40, 82, 83.8, 81.2]

kernels = ['Decision Tree', 'Boosting', 'Neural Net', 'SVM', 'KNN']
temp_x = arange(5)
fig = plt.figure()
plt.style.use('Solarize_Light2')
plt.bar(temp_x - 0.22, letter, width=0.20, color='r', label="Letter")
plt.bar(temp_x + 0.02, fashion, width=0.20, color='b', label="Fashion")
plt.xticks(temp_x, kernels)
plt.xlabel('Algorithms')
plt.ylabel('Best Accuracy')
plt.legend(loc='best')
plt.title('Supervised Learning Algorithms Compared')
fig.savefig('conclusion.png')
plt.close(fig)
