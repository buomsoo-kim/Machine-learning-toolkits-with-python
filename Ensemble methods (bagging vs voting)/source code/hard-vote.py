from sklearn import datasets
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score

data = datasets.load_digits()

X_data = data.images   # load X_data
y_data = data.target   # load y_data

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])    # flatten X_data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 7)    # split data into train & test set

clf_1 = SVC()
clf_2 = SVC()
clf_3 = SVC()

hard_vote_clf = VotingClassifier(estimators = [('svm1', clf_1), ('svm2', clf_2), ('svm3', clf_3)], voting = 'hard')
hard_vote_clf.fit(X_train, y_train)

y_pred = hard_vote_clf.predict(X_test)

print(accuracy_score(y_pred, y_test))