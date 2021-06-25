from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("train0.csv")
test = pd.read_csv("test0.csv")
train_df = train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

test_feature = test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title').values

# fit到BaggingRegressor之中
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)
predictions = bagging_clf.predict(test_feature)
result = pd.DataFrame({"PassengerId": test['PassengerId'],
                       'Survived': predictions.astype(np.int32)})
result.to_csv('result_bagging.csv', index=False)

model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=200, min_child_weight=5, max_depth=7, gamma=0.4)
model.fit(X, y)
predictions = model.predict(test_feature)
result = pd.DataFrame({'PassengerId': test['PassengerId'],
                       'Survived': predictions.astype(np.int32)})
result.to_csv('result_xgb.csv', index=False)


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
alg2 = SVC(probability=True, random_state=29, C=11, gamma=0.05)
rf_clf = RandomForestClassifier()
mv_clf = VotingClassifier(estimators=[('lr', clf), ('xgb', model), ('svc', alg2), ('rf', rf_clf)],
                          voting='hard')
mv_clf.fit(X, y)
predictions = mv_clf.predict(test_feature)
result = pd.DataFrame({'PassengerId': test['PassengerId'],
                       'Survived': predictions.astype(np.int32)})
result.to_csv("result_voting.csv", index=False)

from sklearn.ensemble import StackingClassifier
stacking_clf = StackingClassifier(estimators=[('xgb', model), ('svc', alg2), ('rf', rf_clf)],
                          final_estimator=clf)
stacking_clf.fit(X, y)
predictions = stacking_clf.predict(test_feature)
result = pd.DataFrame({'PassengerId': test['PassengerId'],
                       'Survived': predictions.astype(np.int32)})
result.to_csv('result_stacking.csv', index=False)



