import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_digits

whole_data = load_digits()

X_data = whole_data.images   # load X_data
y_data = whole_data.target   # load y_data

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])    # flatten X_data
y_data = y_data.reshape((y_data.shape[0], 1))

data_merged = np.concatenate((X_data, y_data), axis = 1)
np.random.shuffle(data_merged)
data_merged = data_merged[:300, :]    # use only 300 data instances

X_data = data_merged[:, :-1]
y_data = data_merged[:, -1]

X_scaled = X_data - np.mean(X_data, axis = 0)
std = np.std(X_scaled, axis = 0) + 0.00001    # add a minute number to prevent divide by zero
X_scaled /= std                               # divde by standard deviation

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 7)    # split data into train & test set

clf = LinearSVC(penalty = 'l1', C = 100, dual = False)
clf.fit(X_train, y_train)

y_tr_pred = clf.predict(X_train)
y_te_pred = clf.predict(X_test)

print(accuracy_score(y_tr_pred, y_train))
print(accuracy_score(y_te_pred, y_test))