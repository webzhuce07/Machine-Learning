import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

train = pd.read_csv('cs-training0.csv')
test = pd.read_csv('cs-test0.csv')
print(train.info())

X = train.iloc[:, 2:].values
y = train['SeriousDlqin2yrs'].values

#逻辑回归分类器
params ={"C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
    'penalty': ['l1', 'l2']
}
logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
              random_state=0)
clf = RandomizedSearchCV(estimator=logistic, param_distributions=params, scoring='roc_auc')
search = clf.fit(X, y)
print("Score:", search.best_score_)
print("Score:", search.best_params_)

#决策树
from sklearn.tree import DecisionTreeClassifier
params = {"max_depth": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 40],
       "criterion" : ["gini","entropy"]
}
tree = DecisionTreeClassifier(random_state=0)
clf = RandomizedSearchCV(estimator=tree, param_distributions=params, scoring='roc_auc')
search = clf.fit(X, y)
print("Score:", search.best_score_)
print("Score:", search.best_params_)

#支持向量机
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
params = {'C': [1, 10, 100, 1000],
          'gamma': [0.001, 0.0001],
          'kernel': ['rbf', 'linear'],}
steps = [('scaler', StandardScaler()), ('rbf_svm', SVC())]
pipeline = Pipeline(steps)
clf = RandomizedSearchCV(estimator=pipeline, param_distributions=params, scoring='roc_auc')
search = clf.fit(X, y)
print("Score:", search.best_score_)
print("Score:", search.best_params_)

#支持向量机
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
params = {'C': [1, 10, 100, 1000],
          'gamma': [0.001, 0.0001],
          'kernel': ['rbf', 'linear'],}
steps = [('scaler', StandardScaler()), ('rbf_svm', SVC())]
pipeline = Pipeline(steps)
clf = RandomizedSearchCV(estimator=pipeline, param_distributions=params, scoring='roc_auc')
search = clf.fit(X, y)
print("Score:", search.best_score_)
print("Score:", search.best_params_)

#随机森林
params = {'n_estimators ': [10, 30, 50, 70, 100],
          'criterion  ': ["gini","entropy"],
          'max_depth ': [10, 20, 30]}
rf = RandomForestClassifier()
clf = RandomizedSearchCV(estimator=rf, param_distributions=params, scoring='roc_auc')
search = clf.fit(X, y)
print("Score:", search.best_score_)
print("Score:", search.best_params_)