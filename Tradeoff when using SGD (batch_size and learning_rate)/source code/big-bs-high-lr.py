import numpy as np

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = datasets.load_digits()

X_data = data.images   # load X_data
y_data = data.target   # load y_data
X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])    # flatten X_data

clf = MLPClassifier(hidden_layer_sizes = (25, 25, 25), solver = 'sgd', batch_size = 100, learning_rate_init = 0.01, max_iter = 500) 
clf.fit(X_data, y_data)

y_pred = clf.predict(X_data)
print(accuracy_score(y_pred, y_data))