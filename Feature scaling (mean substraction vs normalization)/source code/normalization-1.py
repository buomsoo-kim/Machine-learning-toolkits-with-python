# normalization - 1

import numpy as np

from sklearn import datasets
from sklearn.model_selection  import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = datasets.load_digits()

X_data = data.images   # load X_data
y_data = data.target   # load y_data

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])    # flatten X_data
X_scaled = X_data - np.mean(X_data, axis = 0)

std = np.std(X_scaled, axis = 0) + 0.00001    # add a minute number to prevent divide by zero
X_scaled /= std    # divde by standard deviation

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.3, random_state = 7)    # split data into train & test set

clf = MLPClassifier(hidden_layer_sizes = (20, 20))   # create a MLP with twenty hidden layers with five neurons
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))