import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('cs-training.csv')

#EDA
#显示所有列
pd.set_option("display.max_columns", None)
#显示所有行
pd.set_option("display.max_rows", None)
print(data_train.head())
print(data_train.dtypes)
print(data_train.describe())
print(data_train.info())
#MonthlyIncome and NumberOfDependents have null

#看下数据有没有重复值
print("Duplicated: ", data_train.duplicated().sum())
#删除重复值
data_train = data_train.drop_duplicates()

data_train.drop('Unnamed: 0', axis=1, inplace=True)

#列名重命名
colnames={'SeriousDlqin2yrs':'Isdlq',
         'RevolvingUtilizationOfUnsecuredLines':'Revol',
         'NumberOfTime30-59DaysPastDueNotWorse':'Num30-59late',
         'NumberOfOpenCreditLinesAndLoans':'Numopen',
         'NumberOfTimes90DaysLate':'Num90late',
         'NumberRealEstateLoansOrLines':'Numestate',
         'NumberOfTime60-89DaysPastDueNotWorse':'Num60-89late',
         'NumberOfDependents':'Numdepend'}
data_train.rename(columns=colnames,inplace=True)

sns.countplot('Isdlq', data=data_train)
badNum = data_train.loc[data_train['Isdlq'] == 1, :].shape[0]
goodNum = data_train.loc[data_train['Isdlq'] == 0, :].shape[0]
print('好坏比：{0}%'.format(round(badNum*100/(goodNum+badNum), 2)))
#data imbalance

#Age数据分布情况
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 6))
sns.distplot(data_train['age'], ax=ax1)
sns.boxplot(y='age', data=data_train, ax=ax2)
#异常值情况
age_mean=data_train['age'].mean()
age_std=data_train['age'].std()
age_lowlimit = age_mean - 3 * age_std
age_uplimit = age_mean + 3 * age_std
print('异常值下限：', age_lowlimit, '异常值上限：', age_uplimit)

age_lowlimitd = data_train.loc[data_train['age'] < age_lowlimit, :]
age_uplimitd = data_train.loc[data_train['age'] > age_uplimit, :]
print('异常值下限比例：{0}%'.format(age_lowlimitd.shape[0]*100/data_train.shape[0]),
     '异常值上限比例：{0}%'.format(age_uplimitd.shape[0]*100/data_train.shape[0]))

data_age = data_train.loc[data_train['age'] > 0, ['age', 'Isdlq']]
data_age.loc[(data_age['age'] > 18) & (data_age['age'] < 40), 'age'] = 1
data_age.loc[(data_age['age'] >= 40) & (data_age['age'] < 60), 'age'] = 2
data_age.loc[(data_age['age'] >= 60) & (data_age['age'] < 80), 'age'] = 3
data_age.loc[(data_age['age'] >= 80), 'age'] = 4
age_Isdlq = data_age.groupby('age')['Isdlq'].sum()
age_total = data_age.groupby('age')['Isdlq'].count()
age_Isratio = age_Isdlq/age_total
plt.figure()
age_Isratio.plot(kind='bar', figsize=(8, 6), color='#4682B4')

#Revol数据分布
figure=plt.figure(figsize=(8,6))
plt.scatter(data_train['Revol'],data_train['age'])
plt.grid()

percent_25 = np.percentile(data_train['Revol'], 25)
percent_75 = np.percentile(data_train['Revol'], 75)
Revol_lowlimit = percent_25-1.5*(percent_75-percent_25)
Revol_uplimit = percent_75+1.5*(percent_75-percent_25)
print('异常值下限值：', Revol_lowlimit, '异常值上限值：', Revol_uplimit)

#将数据分为两部分，小于1和大于1的部分
data1 = data_train.loc[data_train['Revol'] < 1, :]
data2 = data_train.loc[data_train['Revol'] >= 1, :]
#看一下两部分数据分布情况
fig = plt.figure(figsize=(20, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
sns.distplot(data1['Revol'], ax=ax1, bins=10)
sns.distplot(data2['Revol'], ax=ax2, bins=10)

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
sns.distplot(data_train.loc[(data_train['Revol'] >= 1) & (data_train['Revol'] < 100), 'Revol'], bins=10, ax=ax1)
sns.distplot(data_train.loc[(data_train['Revol'] >= 100) & (data_train['Revol'] < 1000), 'Revol'], bins=10, ax=ax2)
sns.distplot(data_train.loc[(data_train['Revol'] >= 1000) & (data_train['Revol'] < 10000), 'Revol'], bins=10, ax=ax3)
sns.distplot(data_train.loc[data_train['Revol'] >= 10000, 'Revol'], bins=10, ax=ax4)

#将区间分为（0-1），(1-10),（10-20），（20-100），（100,1000），（1000-10000），（10000,51000）看一下违约率情况
data_1 = data_train.loc[(data_train['Revol'] >= 0) & (data_train['Revol'] < 1), :]
Is_1 = data_1.loc[data_1['Isdlq'] == 1, :].shape[0]*100/data_1.shape[0]

data_2 = data_train.loc[(data_train['Revol'] >= 1) & (data_train['Revol'] < 10), :]
Is_2 = data_2.loc[data_2['Isdlq'] == 1, :].shape[0]*100/data_2.shape[0]

data_3 = data_train.loc[(data_train['Revol'] >= 10) & (data_train['Revol'] < 20), :]
Is_3 = data_3.loc[data_3['Isdlq'] == 1, :].shape[0]*100/data_3.shape[0]

data_4 = data_train.loc[(data_train['Revol'] >= 20) & (data_train['Revol'] < 100), :]
Is_4=data_4.loc[data_4['Isdlq']==1,:].shape[0]*100/data_4.shape[0]

data_5 = data_train.loc[(data_train['Revol'] >= 100) & (data_train['Revol'] < 1000), :]
Is_5 = data_5.loc[data_5['Isdlq'] == 1, :].shape[0]*100/data_5.shape[0]

data_6 = data_train.loc[(data_train['Revol'] >= 1000) & (data_train['Revol'] < 10000), :]
Is_6 = data_6.loc[data_6['Isdlq'] == 1,:].shape[0]*100/data_6.shape[0]

data_7 = data_train.loc[(data_train['Revol'] >= 10000)&(data_train['Revol'] < 51000), :]
Is_7 = data_7.loc[data_7['Isdlq'] == 1, :].shape[0]*100/data_7.shape[0]

print('0-1违约率为：{0}%'.format(Is_1),
     '1-10违约率为：{0}%'.format(Is_2),
     '10-20违约率为：{0}%'.format(Is_3),
     '20-100违约率为：{0}%'.format(Is_4),
     '100-1000违约率为：{0}%'.format(Is_5),
     '1000-10000违约率为：{0}%'.format(Is_6),
     '10000-51000违约率为：{0}%'.format(Is_7))

#DebtRatio数据的分布情况
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 6))
sns.kdeplot(data_train['DebtRatio'], ax=ax1)
sns.boxplot(y=data_train['DebtRatio'], ax=ax2)

#尝试多次细分
Debt3 = data_train.loc[(data_train['DebtRatio'] >= 1) & (data_train['DebtRatio'] < 1000), :]
Debt4 = data_train.loc[(data_train['DebtRatio'] >= 1) & (data_train['DebtRatio'] < 200), :]
Debt5 = data_train.loc[(data_train['DebtRatio'] >= 1) & (data_train['DebtRatio'] < 10), :]
Debt6 = data_train.loc[(data_train['DebtRatio'] >= 1) & (data_train['DebtRatio'] < 2), :]

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
sns.distplot(Debt3['DebtRatio'], ax=ax1)
sns.distplot(Debt4['DebtRatio'], ax=ax2)
sns.distplot(Debt5['DebtRatio'], ax=ax3)
sns.distplot(Debt6['DebtRatio'], ax=ax4)

#各区间的违约率(0,1),(1-2),(2-10),(10-50),(50-200),(200,1000),1000以上
Debt_1 = data_train.loc[(data_train['DebtRatio'] >= 0) & (data_train['DebtRatio'] < 1), :]
DebIs_1 = Debt_1.loc[Debt_1['Isdlq'] == 1, :].shape[0]*100/Debt_1.shape[0]

Debt_2 = data_train.loc[(data_train['DebtRatio'] >= 1) & (data_train['DebtRatio'] < 2), :]
DebIs_2 = Debt_2.loc[Debt_2['Isdlq'] == 1, :].shape[0]*100/Debt_2.shape[0]

Debt_3 = data_train.loc[(data_train['DebtRatio'] >= 2) & (data_train['DebtRatio'] < 10), :]
DebIs_3 = Debt_3.loc[Debt_3['Isdlq'] == 1, :].shape[0]*100/Debt_3.shape[0]

Debt_4 = data_train.loc[(data_train['DebtRatio'] >= 10) & (data_train['DebtRatio'] < 50), :]
DebIs_4 = Debt_4.loc[Debt_4['Isdlq'] == 1, :].shape[0]*100/Debt_4.shape[0]

Debt_5 = data_train.loc[(data_train['DebtRatio'] >= 50) & (data_train['DebtRatio'] < 200), :]
DebIs_5 = Debt_5.loc[Debt_5['Isdlq']==1,:].shape[0]*100/Debt_5.shape[0]

Debt_6 = data_train.loc[(data_train['DebtRatio'] >= 200) & (data_train['DebtRatio'] < 1000), :]
DebIs_6 = Debt_6.loc[Debt_6['Isdlq'] == 1, :].shape[0]*100/Debt_6.shape[0]

Debt_7 = data_train.loc[data_train['DebtRatio'] >= 1000, :]
DebIs_7 = Debt_7.loc[Debt_7['Isdlq'] == 1, :].shape[0]*100/Debt_7.shape[0]

print('0-1违约率为：{0}%'.format(DebIs_1),
     '1-2违约率为：{0}%'.format(DebIs_2),
     '2-10违约率为：{0}%'.format(DebIs_3),
     '10-50违约率为：{0}%'.format(DebIs_4),
     '50-200违约率为：{0}%'.format(DebIs_5),
     '200-1000违约率为：{0}%'.format(DebIs_6),
     '1000以上违约率为：{0}%'.format(DebIs_7))

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 6))
sns.distplot(data_train['Numopen'], ax=ax1)
sns.boxplot(y=data_train['Numopen'], ax=ax2)
#看一下数据点分布
figure = plt.figure(figsize=(12, 6))
sns.countplot(data_train['Numopen'])
data_train.loc[data_train['Numopen'] > 36, 'Numopen'] = 36
Numopen_dlq = data_train.groupby(['Numopen'])['Isdlq'].sum()
Numopen_total = data_train.groupby(['Numopen'])['Isdlq'].count()
Numopen_dlqratio = Numopen_dlq/Numopen_total
Numopen_dlqratio.plot(kind='bar', figsize=(12, 6), color='#4682B4')

#数据分布
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 6))
sns.distplot(data_train['Numestate'], ax=ax1)
sns.boxplot(y=data_train['Numestate'], ax=ax2)
#大致看一下各数据点数据大小分布
sns.countplot(data_train['Numestate'])
#将大于8的数据和8合并后看一下违约率的情况
data_train.loc[data_train['Numestate']>8,'Numestate']=8
Numestate_dlq = data_train.groupby(['Numestate'])['Isdlq'].sum()
Numestate_total = data_train.groupby(['Numestate'])['Isdlq'].count()
Numestate_dlqratio = Numestate_dlq/Numestate_total
Numestate_dlqratio.plot(kind='bar', figsize=(8, 6), color='#4682B4')

#Numdepend数据分布
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 6))
sns.countplot(data_train['Numdepend'], ax=ax1)
sns.boxplot(y=data_train['Numdepend'], ax=ax2)
D_nullNum = data_train['Numdepend'].isnull().sum()
print('缺失值数量：', D_nullNum, '缺失值比率：{0}%'.format(D_nullNum*100/data_train.shape[0]))

#看一下MonthlyIncome和Numdepend的缺失是否有关联
data_train.loc[(data_train['Numdepend'].isnull()) & (data_train['MonthlyIncome'].isnull()), :].shape[0]

MonthNullDependNot=data_train.loc[(data_train['Numdepend'].notnull())&(data_train['MonthlyIncome']).isnull(), :]
sns.countplot(MonthNullDependNot['Numdepend'])

#MonthlyIncome数据分布
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 6))
sns.kdeplot(data_train['MonthlyIncome'], ax=ax1)
sns.boxplot(y=data_train['MonthlyIncome'], ax=ax2)

#MonthlyIncome缺失值情况
M_nullNum = data_train['MonthlyIncome'].isnull().sum()
print('缺失值数量：', M_nullNum,'缺失值比率：{0}%'.format(M_nullNum*100/data_train.shape[0]))

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(20, 6))
sns.countplot(data_train['Num30-59late'], ax=ax1)
sns.countplot(data_train['Num60-89late'], ax=ax2)
sns.countplot(data_train['Num90late'], ax=ax3)



data_train = datapreprocess.DataPreprocess(data_train)
corr = data_train.corr()
plt.figure(figsize=(14, 2))
sns.heatmap(corr, annot=True, linewidths=.3, cmap='YlGnBu')
#plt.show()

def bin_woe(tar, var, n=None, cat=None):
    """
    连续自变量分箱,woe,iv变换
    tar:target目标变量
    var:进行woe,iv转换的自变量
    n:分组数量
    """
    total_bad = tar.sum()
    total_good = tar.count() - total_bad
    totalRate = total_good / total_bad

    if cat == 's':
        msheet = pd.DataFrame({tar.name: tar, var.name: var, 'var_bins': pd.qcut(var, n, duplicates='drop')})
        grouped = msheet.groupby(['var_bins'])
    elif (cat == 'd') and (n is None):
        msheet = pd.DataFrame({tar.name: tar, var.name: var})
        grouped = msheet.groupby([var.name])

    groupBad = grouped.sum()[tar.name]
    groupTotal = grouped.count()[tar.name]
    groupGood = groupTotal - groupBad
    groupRate = groupGood / groupBad
    groupBadRate = groupBad / groupTotal
    groupGoodRate = groupGood / groupTotal

    woe = np.log(groupRate / totalRate)
    iv = np.sum((groupGood / total_good - groupBad / total_bad) * woe)

    if cat == 's':
        new_var, cut = pd.qcut(var, n, duplicates='drop', retbins=True, labels=woe.tolist())
    elif cat == 'd':
        dictmap = {}
        for x in woe.index:
            dictmap[x] = woe[x]
        new_var, cut = var.map(dictmap), woe.index

    return woe.tolist(), iv, cut, new_var


# 确定变量类型，连续变量还是离散变量
dvar = ['Revol', 'DebtRatio', 'Num30-59late', 'Num60-89late', 'Num90late', 'AllNumlate', 'Withdepend',
        'Numestate', 'Numdepend']
svar = ['MonthlyIncome', 'age', 'Monthlypayment', 'Numopen']


# 可视化woe得分和iv得分
def woe_vs(data):
    cutdict = {}
    ivdict = {}
    woe_dict = {}
    woe_var = pd.DataFrame()
    for var in data.columns:
        if var in dvar:
            woe, iv, cut, new = bin_woe(data['Isdlq'], data[var], cat='d')
            woe_dict[var] = woe
            woe_var[var] = new
            ivdict[var] = iv
            cutdict[var] = cut
        elif var in svar:
            woe, iv, cut, new = bin_woe(data['Isdlq'], data[var], n=5, cat='s')
            woe_dict[var] = woe
            woe_var[var] = new
            ivdict[var] = iv
            cutdict[var] = cut

    ivdict = sorted(ivdict.items(), key=lambda x: x[1], reverse=False)
    iv_vs = pd.DataFrame([x[1] for x in ivdict], index=[x[0] for x in ivdict], columns=['IV'])
    ax = iv_vs.plot(kind='barh',
                    figsize=(12, 12),
                    title='Feature IV',
                    fontsize=10,
                    width=0.8,
                    color='#00688B')
    ax.set_ylabel('Features')
    ax.set_xlabel('IV of Features')

    return ivdict, woe_var, woe_dict, cutdict


# woe转化
ivinfo, woe_data, woe_dict, cut_dict = woe_vs(data_train)

from sklearn.model_selection import train_test_split
IV_info=['Num60-89late','Num90late','AllNumlate','Revol','age']
#X=woe_data[IV_info]
X=data_train.loc[:, IV_info]
y=data_train['Isdlq']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

#Logistic模型建立
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0,
                           solver="sag",
                           penalty="l2",
                           class_weight="balanced",
                           C=1.0,
                           max_iter=500)
model.fit(X_train, y_train)
model_proba = model.predict_proba(X_test)#predict_proba返回的结果是一个数组，包含两个元素，第一个元素是标签为0的概率值，第二个元素是标签为1的概率值
model_score=model_proba[:,1]

#绘制ROC曲线，计算AUC值
from sklearn.metrics import roc_curve, roc_auc_score
fpr,tpr,thresholds =roc_curve(y_test,model_score)
auc_score=roc_auc_score(y_test,model_score)
plt.plot(fpr, tpr, linewidth=2, label='AUC = %0.2f'%auc_score)
plt.plot([0,1],[0,1], "k--")
plt.axis([0,1,0,1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
