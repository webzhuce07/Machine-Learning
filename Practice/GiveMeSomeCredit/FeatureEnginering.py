import warnings
warnings.filterwarnings('ignore')
import pandas as pd

train = pd.read_csv('cs-training.csv')
test = pd.read_csv('cs-test.csv')

#处理缺失值
test['NumberOfDependents'].fillna(0, inplace=True)
train['NumberOfDependents'].fillna(0, inplace=True)
test['MonthlyIncome'].fillna(train['MonthlyIncome'].mean(), inplace=True)
train['MonthlyIncome'].fillna(train['MonthlyIncome'].mean(), inplace=True)

#分箱处理
train.loc[(train['age'] >= 18) & (train['age'] < 40), 'age_range'] = 1
train.loc[(train['age'] >= 40) & (train['age'] < 60), 'age_range'] = 2
train.loc[(train['age'] >= 60) & (train['age'] < 80), 'age_range'] = 3
train.loc[(train['age'] >= 80), 'age_range'] = 4
train.loc[(train['age'] < 18), 'age_range'] = 4

test.loc[(test['age'] >= 18) & (test['age'] < 40), 'age_range'] = 1
test.loc[(test['age'] >= 40) & (test['age'] < 60), 'age_range'] = 2
test.loc[(test['age'] >= 60) & (test['age'] < 80), 'age_range'] = 3
test.loc[(test['age'] >= 80), 'age_range'] = 4
test.loc[(test['age'] < 18), 'age_range'] = 4

from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
selected_feature = SelectFromModel(XGBClassifier()).fit(train.iloc[:, 2:], train['SeriousDlqin2yrs'])
print(selected_feature)
print(selected_feature.get_support())
train_x = train.iloc[:, 2:]
print(train_x.columns[selected_feature.get_support()])

from matplotlib import pyplot
from xgboost import plot_importance
model_XGB = XGBClassifier()
model_XGB.fit(train_x, train['SeriousDlqin2yrs'])
pyplot.bar(range(len(model_XGB.feature_importances_)), model_XGB.feature_importances_)
plot_importance(model_XGB)
pyplot.show()

train.to_csv('cs-training0.csv', index=False)
test.to_csv('cs-test0.csv', index=False)
