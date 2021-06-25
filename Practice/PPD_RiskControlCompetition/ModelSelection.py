from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd

train_master_ = pd.read_csv("master.csv")
X_train = train_master_.drop(['target'], axis=1)
X_train = StandardScaler().fit_transform(X_train)
y_train = train_master_['target']
print(X_train.shape, y_train.shape)

# 使用StratifiedKFold分层采样，保证预测target的分布合理，并且shuffle随机
cv = StratifiedKFold(n_splits=3, shuffle=True)

# 计算auc accuracy recall
def estimate(estimator, name='estimator'):
    auc = cross_val_score(estimator, X_train, y_train, scoring='roc_auc', cv=cv).mean()
    accuracy = cross_val_score(estimator, X_train, y_train, scoring='accuracy', cv=cv).mean()
    recall = cross_val_score(estimator, X_train, y_train, scoring='recall', cv=cv).mean()

    print("{}: auc:{:f}, recall:{:f}, accuracy:{:f}".format(name, auc, recall, accuracy))

#     skplt.plot_learning_curve(estimator, X_train, y_train)
#     plt.show()

#     estimator.fit(X_train, y_train)
#     y_probas = estimator.predict_proba(X_train)
#     skplt.plot_roc_curve(y_true=y_train, y_probas=y_probas)
#     plt.show()


estimate(XGBClassifier(learning_rate=0.1, n_estimators=20, objective='binary:logistic'), 'XGBClassifier')
estimate(RidgeClassifier(), 'RidgeClassifier')
estimate(LogisticRegression(), 'LogisticRegression')
# estimate(RandomForestClassifier(), 'RandomForestClassifier')
estimate(AdaBoostClassifier(), 'AdaBoostClassifier')
# estimate(SVC(), 'SVC')# too long to wait
# estimate(LinearSVC(), 'LinearSVC')

# XGBClassifier: auc:0.747668, recall:0.000000, accuracy:0.944575
# RidgeClassifier: auc:0.754218, recall:0.000000, accuracy:0.944433
# LogisticRegression: auc:0.758454, recall:0.015424, accuracy:0.942010
# AdaBoostClassifier: auc:0.784086, recall:0.013495, accuracy:0.943791
