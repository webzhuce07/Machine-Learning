import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei']

train = pd.read_csv('cs-training.csv')
test = pd.read_csv('cs-test.csv')
#了解缺失值情况和每列的数据类型
print(train.info())
print(test.info())
#MonthlyIncome NumberOfDependents

null_data = train.isnull().any()
test_null_data = test.isnull().any()
print(null_data, test_null_data)

#方差过小，说明信息冗余
print(train.describe())
print(test.describe())

#看下数据有没有重复值
print(train.duplicated().sum())
#删除重复值
train = train.drop_duplicates()

print(train.columns)
# #train target 分布
# plt.figure()
# sns.countplot(train['SeriousDlqin2yrs'])
# plt.title('逾期情况（1为逾期）')
# plt.ylabel('人数')
# plt.show()
#
# plt.figure()
# sns.boxplot(y=train['RevolvingUtilizationOfUnsecuredLines'])
# plt.show()
#
# plt.figure()
# train_0 = train[train['SeriousDlqin2yrs'] == 0]
# train_1 = train[train['SeriousDlqin2yrs'] == 1]
# normal_0 = train_0[train_0['RevolvingUtilizationOfUnsecuredLines'] <= 1.0]
# normal_1 = train_1[train_1['RevolvingUtilizationOfUnsecuredLines'] <= 1.0]
# sns.distplot([normal_0['RevolvingUtilizationOfUnsecuredLines']], bins=5, label='非逾期')
# sns.distplot(normal_1['RevolvingUtilizationOfUnsecuredLines'], bins=5, label='逾期')
# plt.title('逾期与否与总余额比值的直方图')
#
# normal_0 = train_0[train_0['RevolvingUtilizationOfUnsecuredLines'] <= 1.0]
# normal_1 = train_1[train_1['RevolvingUtilizationOfUnsecuredLines'] <= 1.0]
#
# #age
# fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 6))
# sns.distplot(train['age'], bins=4, ax=ax1)
# sns.boxplot(y=train['age'], ax=ax2)
# plt.show()
#
# #age range 与 逾期的关系
# plt.figure()
# data_age = train.loc[:, ['age', 'SeriousDlqin2yrs']]
# data_age.loc[(data_age['age'] >= 18) & (data_age['age'] < 40), 'age_range'] = 1
# data_age.loc[(data_age['age'] >= 40) & (data_age['age'] < 60), 'age_range'] = 2
# data_age.loc[(data_age['age'] >= 60) & (data_age['age'] < 80), 'age_range'] = 3
# data_age.loc[(data_age['age'] >= 80), 'age_range'] = 0
# data_age.loc[(data_age['age'] < 18), 'age_range'] = 0
# data_0 = data_age.age_range[data_age.SeriousDlqin2yrs == 0].value_counts()
# data_1 = data_age.age_range[data_age.SeriousDlqin2yrs == 1].value_counts()
# df = pd.DataFrame({u'逾期': data_1, u'非预期': data_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各年龄段的逾期情况")
# plt.xlabel(u"年龄段")
# plt.ylabel(u"人数")
# plt.show()
#
# plt.figure()
# age_Isdlq = data_age.groupby('age_range')['SeriousDlqin2yrs'].sum()
# age_total = data_age.groupby('age_range')['SeriousDlqin2yrs'].count()
# age_Isratio = age_Isdlq/age_total
# age_Isratio.plot(kind='bar', figsize=(8, 6), color='#4682B4')
# plt.show()


plt.figure()
corr = np.corrcoef(train, rowvar=False)
sns.heatmap(train.astype(float).corr(), annot=True)
plt.show()


