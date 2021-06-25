import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei']

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#了解缺失值情况和每列的数据类型
print(train.info())
print(test.info())
#方差过小，说明信息冗余
print(train.describe())
print(test.describe())

#乘客获救情况分布
plt.figure()
sns.countplot(train['Survived'])
plt.title('获救情况（1为获救）')
plt.ylabel('人数')

#乘客Pclass属性
plt.figure()
sns.countplot(train['Pclass'])
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.figure()
Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
print(train[['Pclass', 'Survived']].groupby('Pclass').mean())
#Pclass == 1获救概率大些

#乘客name属性


#乘客age属性
plt.figure()
plt.scatter(train.Survived, train.Age)
plt.ylabel(u"年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")

plt.figure()
Survived_m = train.Survived[train.Sex == 'male'].value_counts()
Survived_f = train.Survived[train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.show()
print(train[['age', 'Survived']].groupby('age').mean())

#乘客Embarked属性
plt.figure()
train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
plt.show()

plt.figure()
train.Age[train.Pclass == 1].plot(kind='kde')
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
# plots an axis lable
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
# sets our legend for our graph.
plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')
plt.show()

fig = plt.figure()
plt.title(u"根据舱等级和性别的获救情况")
ax1 = fig.add_subplot(141)
female_high = train.Survived[train.Sex == 'female'][train.Pclass != 3].value_counts()
female_high.plot(kind='bar', label='female highclass', color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
female_low = train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts()
female_low.plot(kind='bar',  label='female, low class', color='pink')

ax3 = fig.add_subplot(143, sharey=ax1)
male_high = train.Survived[train.Sex == 'male'][train.Pclass != 3].value_counts()
male_high.plot(kind='bar', label='male, high class', color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
male_low = train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts()
male_low.plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')
plt.show()

Survived_0 = train.Embarked[train.Survived == 0].value_counts()
Survived_1 = train.Embarked[train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")
plt.show()

g = train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)

g = train.groupby(['Parch', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)

#统计Carbin是否缺失与获救结果的关系
fig = plt.figure()
Survived_cabin = train.Survived[pd.notnull(train.Cabin)].value_counts()
Survived_nocabin = train.Survived[pd.isnull(train.Cabin)].value_counts()
df = pd.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无")
plt.ylabel(u"人数")

train.loc[(train.Cabin.notnull()), 'cabin_flag'] = "Yes"
train.loc[(train.Cabin.isnull()), 'cabin_flag'] = "No"
print(train[['cabin_flag', 'Survived']].groupby('cabin_flag').mean())

plt.show()
