from sklearn.model_selection  import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = datasets.load_digits()

X_data = data.images   # load X_data
y_data = data.target   # load y_data

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])    # flatten X_data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 7)    # split data into train & test set

k = 10

clf = SVC()    # create a SVM classifier

kfold = KFold(n_splits = k, random_state = 777)

results = cross_val_score(clf, X_train, y_train, cv = kfold)

print('Accuracy Score')
print('Avearge: ', results.mean())
print('Standard deviation: ', results.std())