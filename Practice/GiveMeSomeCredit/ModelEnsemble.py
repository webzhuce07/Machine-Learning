import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

train = pd.read_csv('cs-training0.csv')
test = pd.read_csv('cs-test0.csv')

X = train.iloc[:, 2:].values
y = train['SeriousDlqin2yrs'].values
test_feature = test.iloc[:, 2:].values

svc = SVC()
pipeline = Pipeline([('scaler', StandardScaler()), ('SVM', svc)])
rf = RandomForestClassifier()
tree = DecisionTreeClassifier()
lr = LogisticRegression()

stacking_clf = StackingClassifier(estimators=[('tree', tree), ('svc', pipeline), ('rf', rf)],
                          final_estimator=lr)
rf.fit(X, y)
predictions = rf.predict_proba(test_feature)
#predict_proba返回的结果是一个数组，包含两个元素，第一个元素是标签为0的概率值，第二个元素是标签为1的概率值
result = pd.DataFrame({'Id': test.iloc[:, 0],
                       'Probability': predictions[:, 1].astype(np.float64)})
result.to_csv('result_stacking.csv', index=False)