import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']#设置图表的标题可以为中文
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv("cs-training.csv")
print(train.head(10))
#将第一列数据作为数据框的行索引
# train.set_index(["Unnamed: 0"], inplace=True)
# train.to_csv("my_new_file.csv", index=None)
print(train.head(10))
#数据有多少行
print(train.shape[0])
#看下数据有没有重复值
print(train.duplicated().sum())
#删除重复值
train = train.drop_duplicates()
#EDA
train_y0 = train[train['SeriousDlqin2yrs'] == 0]
train_y1 = train[train['SeriousDlqin2yrs'] == 1]
line0 = train_y0.RevolvingUtilizationOfUnsecuredLines.sort_values(ascending=False)
#line1 = train.iloc[:, 1].sort_values(ascending=False)
line1 = train_y1.RevolvingUtilizationOfUnsecuredLines.sort_values(ascending=False)
binsVal = np.arange(0, 2, 0.2)
plt.subplot(2, 1, 1)
plt.hist(line0, binsVal, facecolor="green", edgecolor="black", alpha=0.3)
plt.xlabel("总余额比值区间")
plt.ylabel("非逾期组频率")
plt.title('逾期与否与总余额比值的直方图')
plt.subplot(2, 1, 2)
plt.hist(line1, binsVal, facecolor="green", edgecolor="black", alpha=0.3)
plt.ylabel('逾期组频数')
plt.xlabel('总余额比值区间')
#plt.show()

fig, (axis1, axis2) = plt.subplots(1, 2)
sns.barplot

plt.figure()
line0 = train_y0['NumberOfTime30-59DaysPastDueNotWorse'].sort_values(ascending=False)
line1 = train_y1['NumberOfTime30-59DaysPastDueNotWorse'].sort_values(ascending=False)
binsVal = np.arange(0, 7, 1)
plt.subplot(2, 1, 1)
plt.hist(line0, binsVal, facecolor='green', edgecolor='black', alpha=0.3)
plt.xlabel("逾期30至59天的次数区间")
plt.ylabel("非逾期组频率")
plt.title('逾期与否与逾期30至59天的次数的直方图')
plt.subplot(2, 1, 2)
plt.hist(line1, binsVal, facecolor='green', edgecolor='black', alpha=0.3)
plt.xlabel("逾期30至59天的次数区间")
plt.ylabel("逾期组频率")
#plt.show()

plt.figure()
sns.distplot(train["age"], color="black")#绘制年龄的直方图
#plt.show()

plt.figure()
bins = [20, 30, 40, 50, 60, max(train.age) + 1]
labels=["20-30岁","30-40岁","40-50岁","50-60岁","60岁以上"]
train["年龄分层"] = pd.cut(train.age, bins=bins, labels=labels)
trainsage_groupby = train.pivot_table(values=["age"], index=["年龄分层"], columns=["SeriousDlqin2yrs"], aggfunc=[np.size])
trainsage_groupby["逾期占比"]=trainsage_groupby.iloc[:, 1]/(trainsage_groupby.iloc[:, 0]+trainsage_groupby.iloc[:, 1])
trainsage_groupby["合计人数"]=(trainsage_groupby.iloc[:, 0]+trainsage_groupby.iloc[:, 1])

n1 = []
n2 = []
i = j= 0
for i in range(5):
    n1.append(trainsage_groupby.iloc[i, 0])
    n2.append(trainsage_groupby.iloc[i, 1])
tup1 = tuple(n1)
tup2 = tuple(n2)

n_groups = 5#组数
fig, ax = plt.subplots()#条形图
index = np.arange(n_groups)
bar_width = 0.35#条宽
opacity = 0.6#颜色深浅
error_config = {"ecolor":"0.1"}#误差线透明度
rectsl = plt.bar(index,tup1,bar_width,alpha=opacity,color="g",label="非逾期组")#第一类
rects2 = plt.bar(index+bar_width,tup2,bar_width,alpha=opacity,color="r",label="逾期组")#第二类条
plt.xlabel("年龄分组")
plt.ylabel("人数")
plt.title("逾期与否在各年龄分组下的人数")
plt.xticks(index+bar_width,("20-30岁","30-40岁","40-50岁","50-60岁","60岁以上"))
plt.legend()
plt.tight_layout()
plt.show()

