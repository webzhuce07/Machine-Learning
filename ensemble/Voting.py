import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
##加载模型
from sklearn.datasets import load_iris #加载数据
from sklearn.model_selection import train_test_split #切分训练集与观测集
from sklearn.preprocessing import StandardScaler  #标准化数据
from sklearn.preprocessing import LabelEncoder #标准化分类变量

##初步处理数据
iris = load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=1, stratify=y)

##我们使用训练集训练三种不同的分类器：逻辑回归+决策树+k近邻分类
from sklearn.model_selection import cross_val_score #10折交叉验证评价模型
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline   # 管道简化工作流
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe1 = Pipeline([('SC', StandardScaler()), ('clf', clf1)])
pipe3 = Pipeline([('sc', StandardScaler()), ('cf', clf3)])
clf_labels = ['Logistic regression', 'Decision tree', 'KNN']
print('10-folds cross validation :\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10,
                             scoring="roc_auc")
    print("ROC AUC: %0.2f(+/- %0.2f[%s])" %(scores.mean(), scores.std(), label))

## 我们使用MajorityVoteClassifier集成：
from sklearn.ensemble import VotingClassifier
mv_clf = VotingClassifier(estimators=[('pipe1', pipe1), ('clf2', clf2), ('pipe3', pipe3)],
                          voting='soft')
clf_labels += ['MajorityVoteClassifier']
all_clf = [pipe1, clf2, pipe3, mv_clf]
print('10-folds cross validation :\n')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10,
                             scoring='roc_auc')
    print("ROC AUC: %0.2f(+/- %0.2f)[%s]" %(scores.mean(), scores.std(), label))

## 使用ROC曲线评估集成分类器
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
plt.figure(figsize=(10, 6))
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s(auc=%0.2f)'
             %(label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False positive rate(RPR)')
plt.ylabel('True positive rate(TPR')
plt.show()
