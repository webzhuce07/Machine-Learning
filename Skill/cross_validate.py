from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
print("All data:", iris.data.shape, iris.target.shape)
X_train, X_test, y_train, y_test = train_test_split( iris.data, iris.target,
                                                     test_size=0.4, random_state=0)
print("Train: ", X_train.shape, y_train.shape)
print("Test: ", X_test.shape, y_test.shape)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("score:", score)

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("cross validate score:", scores)
