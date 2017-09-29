import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, auc, average_precision_score
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

data = datasets.load_breast_cancer()

X_data = data.data
y_data = data.target    

# original data has 212 '0' (malignant) and 357 '1' (benign) class labels
counts = Counter(y_data)
print(counts)     

y_data = y_data.reshape((y_data.shape[0], 1))

data_merged = np.concatenate((X_data, y_data), axis = 1)
data_merged = data_merged[data_merged[:, -1].argsort()]

imbal_data = data_merged[50:]

X_data = imbal_data[:, :-1]
y_data = imbal_data[:, -1]

# now, we have a slightly more unbalanced dataset of about 2:1
counts = Counter(y_data)
print(counts)    

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 7)    # split data into train & test set

clf = RandomForestClassifier() 
clf.fit(X_train, y_train)

y_score = clf.predict_proba(X_test)    # predict probabilities of each data instance to calculate the ROC curve
y_score = np.array(y_score[:, -1], dtype = np.float32)
print(roc_auc_score(y_test, y_score))

fpr, tpr, _ = roc_curve(y_test, y_score)

plt.plot(fpr, tpr, color= 'red', lw = 2)
plt.plot([0, 1], [0, 1], color='navy', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()