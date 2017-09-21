import numpy as np
import scipy

from sklearn import datasets
from sklearn.model_selection  import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

data = datasets.load_digits()

X_data = data.images   # load X_data
y_data = data.target   # load y_data

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])    # flatten X_data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 7)    # split data into train & test set

clf = MLPClassifier(hidden_layer_sizes = (5, 5))   # create a MLP with two hidden layers with five neurons

# assign parameters for random search
activation = ['tanh', 'relu']
solver = ['sgd', 'adam']
batch_size = randint(50, 100)
learning_rate_init = uniform(0.001, 0.1)
max_iter = randint(200, 500)

# create a dictionary to carry parameter settings
parameters = dict(activation = activation, solver = solver, batch_size = batch_size, learning_rate_init = learning_rate_init, max_iter = max_iter)
searcher = RandomizedSearchCV(estimator = clf, param_distributions = parameters, cv = 5, n_iter = 96, scoring = 'accuracy')
random_result = searcher.fit(X_train, y_train)

# print out accuracy results and model setting
print("Best model: %s with accuracy of %f" % (random_result.best_params_, random_result.best_score_))
for params, mean_score, scores in random_result.grid_scores_:
    print("MEAN: %f (STD: %f) with: %r" % (scores.mean(), scores.std(), params))