import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set(style='whitegrid')

# 用arrow lib，把日期解析成年、月、日、周、星期几、月初/月中/月末。带入模型前进行one-hot encoding
def parse_date(date_str, str_format='YYYY/MM/DD'):
    d = pd.to_datetime(date_str, format='%Y-%m-%d')
    # 月初，月中，月末
    month_stage = int((d.day-1) / 10) + 1
    # return (d.value, d.year, d.month, d.day, d.week, d.isoweekday(), month_stage)
    # return (int(d.value), d.year, d.month, d.day, d.week, d.isoweekday(), month_stage)
    return (0, 0, 0, 0, 0, 0, 0)
print(parse_date('2014/3/20', 'YYYY/M/D'))

# 显示列名
def show_cols(df):
    for c in df.columns:
        print(c)


train_master = pd.read_csv('./Training Set/PPD_Training_Master_GBK_3_1_Training_Set.csv', encoding='gbk')
train_loginfo = pd.read_csv('./Training Set/PPD_LogInfo_3_1_Training_Set.csv', encoding='gbk')
train_userinfo = pd.read_csv('./Training Set/PPD_Userupdate_Info_3_1_Training_Set.csv', encoding='gbk')

# train_master中每一列的NULL值数量
null_sum = train_master.isnull().sum()
# train_master中每一列的NULL值数量不为0的
null_sum = null_sum[null_sum!=0]
null_sum_df = DataFrame(null_sum, columns=['num'])
# 缺失率
null_sum_df['ratio'] = null_sum_df['num'] / 30000.0
null_sum_df.sort_values(by='ratio', ascending=False, inplace=True)
print(null_sum_df.head(10))

# 删除缺失严重的列
train_master.drop(['WeblogInfo_3', 'WeblogInfo_1', 'UserInfo_11', 'UserInfo_13', 'UserInfo_12', 'WeblogInfo_20'],
                  axis=1, inplace=True)

# 删除缺失严重的行
record_nan = train_master.isnull().sum(axis=1).sort_values(ascending=False)
print(record_nan.head())
# 删除缺失数量>=5的行
drop_record_index = [i for i in record_nan.loc[(record_nan >= 5)].index]
# 删除之前(30000, 222)
print('before train_master shape {}'.format(train_master.shape))
train_master.drop(drop_record_index, inplace=True)
# 删除之后(29189, 222)
print('after train_master shape {}'.format(train_master.shape))
# len(drop_record_index)

# 所有nan值的数量
print('before all nan num: {}'.format(train_master.isnull().sum().sum()))

# UserInfo_2为null的行，UserInfo_2列置为'位置地点'
train_master.loc[train_master['UserInfo_2'].isnull(), 'UserInfo_2'] = '位置地点'
# UserInfo_4为null的行，UserInfo_4列置为'位置地点'
train_master.loc[train_master['UserInfo_4'].isnull(), 'UserInfo_4'] = '位置地点'

def fill_nan(f, method):
    if method == 'most':
        # 填充为最高频
        common_value = pd.value_counts(train_master[f], ascending=False).index[0]
    else:
        # 填充为均值
        common_value = train_master[f].mean()
    train_master.loc[train_master[f].isnull(), f] = common_value

# 通过pd.value_counts(train_master[f])的观察得到经验
fill_nan('UserInfo_1', 'most')
fill_nan('UserInfo_3', 'most')
fill_nan('WeblogInfo_2', 'most')
fill_nan('WeblogInfo_4', 'mean')
fill_nan('WeblogInfo_5', 'mean')
fill_nan('WeblogInfo_6', 'mean')
fill_nan('WeblogInfo_19', 'most')
fill_nan('WeblogInfo_21', 'most')

print('after all nan num: {}'.format(train_master.isnull().sum().sum()))


ratio_threshold = 0.5
binarized_features = []
binarized_features_most_freq_value = []

# 不同period的third_party_feature均值汇总在一起，结果并不好，故取消
# third_party_features = []

# 遍历所有列，除去target之外的列进行如下处理
for f in train_master.columns:
    if f in ['target']:
        continue
    # 非空值的数量
    not_null_sum = (train_master[f].notnull()).sum()
    # 特征取值最多的index
    most_count = pd.value_counts(train_master[f], ascending=False).iloc[0]
    # 特征取值最多的值
    most_value = pd.value_counts(train_master[f], ascending=False).index[0]
    # 特征取值最多的值占所有取值的比率
    ratio = most_count / not_null_sum
    # 如果大于阈值则归入二值化的特征中
    if ratio > ratio_threshold:
        binarized_features.append(f)
        binarized_features_most_freq_value.append(most_value)

# 数值型特征（除去类型为object，除去'Idx', 'target'，除去binarized_features）
numerical_features = [f for f in train_master.select_dtypes(exclude = ['object']).columns
                      if f not in(['Idx', 'target']) and f not in binarized_features]

# 类别型特征（包括类型为object，除去'Idx', 'target'，除去binarized_features）
categorical_features = [f for f in train_master.select_dtypes(include = ["object"]).columns
                        if f not in(['Idx', 'target']) and f not in binarized_features]

# 遍历所有二值化特征，加名称前缀b_，将最多的取值置为0，其余的置为1
for i in range(len(binarized_features)):
    f = binarized_features[i]
    most_value = binarized_features_most_freq_value[i]
    train_master['b_' + f] = 1
    train_master.loc[train_master[f] == most_value, 'b_' + f] = 0
    train_master.drop([f], axis=1, inplace=True)

feature_unique_count = []
# 遍历数值型特征，统计各个特征取值不为0的数量
for f in numerical_features:
    feature_unique_count.append((np.count_nonzero(train_master[f].unique()), f))

# print(sorted(feature_unique_count))

# 遍历，将取值数量<=10的归为类别型特征
for c, f in feature_unique_count:
    if c <= 10:
        print('{} moved from numerical to categorical'.format(f))
        numerical_features.remove(f)
        categorical_features.append(f)

melt = pd.melt(train_master, id_vars=['target'], value_vars = [f for f in numerical_features])
print(melt.head(50))
print(melt.shape)
g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
g.map(sns.stripplot, 'target', 'value', jitter=True, palette="muted")

# Seaborn画图，查看正负样本中特征取值的分布，删除离群值

print('{} lines before drop'.format(train_master.shape[0]))

train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_1 > 250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period6_2 > 400].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_2 > 250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period6_3 > 2000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_3 > 1250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period6_4 > 1500].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_4 > 1250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_5 > 400)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_7 > 2000)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_6 > 1500)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_5 > 1000) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_8 > 1500)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_8 > 1000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_16 > 2000000) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_14 > 1000000) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_12 > 60)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_11 > 120) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_11 > 20) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_13 > 200000)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_13 > 150000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_15 > 40000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_17 > 130000) & (train_master.target == 0)].index, inplace=True)


train_master.drop(train_master[train_master.ThirdParty_Info_Period5_1 > 500].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_2 > 500].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_3 > 3000) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_3 > 2000)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_5 > 500].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_4 > 2000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_6 > 700].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_6 > 300) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_7 > 4000)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_8 > 800)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_11 > 200)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_13 > 200000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_14 > 150000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_15 > 75000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_16 > 180000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_17 > 150000].index, inplace=True)

# go above

train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_1 > 400)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_2 > 350)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_3 > 1500)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_4 > 1600].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_4 > 1250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_5 > 500].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_6 > 800].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_6 > 400) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_8 > 1000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_13 > 250000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_14 > 200000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_15 > 70000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_16 > 210000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_17 > 160000].index, inplace=True)


train_master.drop(train_master[train_master.ThirdParty_Info_Period3_1 > 400].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_2 > 380].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_3 > 1750].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_4 > 1750].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period3_4 > 1250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_5 > 600].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_6 > 800].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period3_6 > 400) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period3_7 > 1600) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_8 > 1000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_13 > 300000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_14 > 200000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_15 > 80000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_16 > 300000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_17 > 150000].index, inplace=True)


train_master.drop(train_master[train_master.ThirdParty_Info_Period2_1 > 400].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_1 > 300) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_2 > 400].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_2 > 300) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_3 > 1800].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_3 > 1500) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_4 > 1500].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_5 > 580].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_6 > 800].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_6 > 400) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_7 > 2100].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_8 > 700) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_11 > 120].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_13 > 300000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_14 > 170000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_15 > 80000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_15 > 50000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_16 > 300000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_17 > 150000].index, inplace=True)


train_master.drop(train_master[train_master.ThirdParty_Info_Period1_1 > 350].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_1 > 200) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_2 > 300].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_2 > 190) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_3 > 1500].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_4 > 1250].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_5 > 400].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_6 > 500].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_6 > 250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_7 > 1800].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_8 > 720].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_8 > 600) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_11 > 100].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_13 > 200000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_13 > 140000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_14 > 150000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_15 > 70000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_15 > 30000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_16 > 200000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_16 > 100000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_17 > 100000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_17 > 80000) & (train_master.target == 1)].index, inplace=True)

train_master.drop(train_master[train_master.WeblogInfo_4 > 40].index, inplace=True)
train_master.drop(train_master[train_master.WeblogInfo_6 > 40].index, inplace=True)
train_master.drop(train_master[train_master.WeblogInfo_7 > 150].index, inplace=True)
train_master.drop(train_master[train_master.WeblogInfo_16 > 50].index, inplace=True)
train_master.drop(train_master[(train_master.WeblogInfo_16 > 25) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.WeblogInfo_17 > 100].index, inplace=True)
train_master.drop(train_master[(train_master.WeblogInfo_17 > 80) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.UserInfo_18 < 10].index, inplace=True)

print('{} lines after drop'.format(train_master.shape[0]))

# melt = pd.melt(train_master, id_vars=['target'], value_vars = [f for f in numerical_features if f != 'Idx'])
g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
g.map(sns.distplot, "value")

# train_master_log = train_master.copy()
numerical_features_log = [f for f in numerical_features if f not in ['Idx']]

# 将数值型特征取log
for f in numerical_features_log:
    train_master[f + '_log'] = np.log1p(train_master[f])
    train_master.drop([f], axis=1, inplace=True)

from math import inf

(train_master == -inf).sum().sum()
train_master.replace(-inf, -1, inplace=True)

# log后的密度图，应该分布靠近正态分布了
melt = pd.melt(train_master, id_vars=['target'], value_vars = [f+'_log' for f in numerical_features])
g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
g.map(sns.distplot, "value")

# log后的分布图，看是否有log后的outlier
g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
g.map(sns.stripplot, 'target', 'value', jitter=True, palette="muted")


melt = pd.melt(train_master, id_vars=['target'], value_vars=[f for f in categorical_features])
g = sns.FacetGrid(melt, col='variable', col_wrap=4, sharex=False, sharey=False)
g.map(sns.countplot, 'value', palette="muted")
target_corr = np.abs(train_master.corr()['target']).sort_values(ascending=False)
print(target_corr)

# at_home，猜测UserInfo_2和UserInfo_8可能表示用户的当前居住地和户籍地，从而判断用户是否在老家。
train_master['at_home'] = np.where(train_master['UserInfo_2']==train_master['UserInfo_8'], 1, 0)

train_master_ = train_master.copy()

def parse_ListingInfo(date):
    d = parse_date(date, 'YYYY/M/D')
    return Series(d,
                  index=['ListingInfo_timestamp', 'ListingInfo_year', 'ListingInfo_month',
                           'ListingInfo_day', 'ListingInfo_week', 'ListingInfo_isoweekday', 'ListingInfo_month_stage'],
                  dtype=np.int32)

ListingInfo_parsed = train_master_['ListingInfo'].apply(parse_ListingInfo)
print('before train_master_ shape {}'.format(train_master_.shape))
train_master_ = train_master_.merge(ListingInfo_parsed, how='left', left_index=True, right_index=True)
print('after train_master_ shape {}'.format(train_master_.shape))


def loginfo_aggr(group):
    # group的数量
    loginfo_num = group.shape[0]
    # 操作代码的数量
    loginfo_LogInfo1_unique_num = group['LogInfo1'].unique().shape[0]
    # 登录时间的数量
    loginfo_active_day_num = group['LogInfo3'].unique().shape[0]
    # 处理登录时间最小值
    min_day = parse_date(np.min(group['LogInfo3']), str_format='YYYY-MM-DD')
    # 处理登陆时间最大值
    max_day = parse_date(np.max(group['LogInfo3']), str_format='YYYY-MM-DD')
    # 最大值和最小值相差多少天
    gap_day = round((max_day[0] - min_day[0]) / 86400)

    indexes = {
        'loginfo_num': loginfo_num,
        'loginfo_LogInfo1_unique_num': loginfo_LogInfo1_unique_num,
        'loginfo_active_day_num': loginfo_active_day_num,
        'loginfo_gap_day': gap_day,
        'loginfo_last_day_timestamp': max_day[0]
    }

    # TODO every individual LogInfo1,LogInfo2 count

    def sub_aggr_loginfo(sub_group):
        return sub_group.shape[0]

    sub_group = group.groupby(by=['LogInfo1', 'LogInfo2']).apply(sub_aggr_loginfo)
    indexes['loginfo_LogInfo12_unique_num'] = sub_group.shape[0]
    return Series(data=[indexes[c] for c in indexes], index=[c for c in indexes])


train_loginfo_grouped = train_loginfo.groupby(by=['Idx']).apply(loginfo_aggr)
train_loginfo_grouped.head()

train_loginfo_grouped.to_csv('train_loginfo_grouped.csv', header=True, index=True)
train_loginfo_grouped = pd.read_csv('train_loginfo_grouped.csv')
train_loginfo_grouped.head()


def userinfo_aggr(group):
    op_columns = ['_EducationId', '_HasBuyCar', '_LastUpdateDate',
                  '_MarriageStatusId', '_MobilePhone', '_QQ', '_ResidenceAddress',
                  '_ResidencePhone', '_ResidenceTypeId', '_ResidenceYears', '_age',
                  '_educationId', '_gender', '_hasBuyCar', '_idNumber',
                  '_lastUpdateDate', '_marriageStatusId', '_mobilePhone', '_qQ',
                  '_realName', '_regStepId', '_residenceAddress', '_residencePhone',
                  '_residenceTypeId', '_residenceYears', '_IsCash', '_CompanyPhone',
                  '_IdNumber', '_Phone', '_RealName', '_CompanyName', '_Age',
                  '_Gender', '_OtherWebShopType', '_turnover', '_WebShopTypeId',
                  '_RelationshipId', '_CompanyAddress', '_Department',
                  '_flag_UCtoBcp', '_flag_UCtoPVR', '_WorkYears', '_ByUserId',
                  '_DormitoryPhone', '_IncomeFrom', '_CompanyTypeId',
                  '_CompanySizeId', '_companyTypeId', '_department',
                  '_companyAddress', '_workYears', '_contactId', '_creationDate',
                  '_flag_UCtoBCP', '_orderId', '_phone', '_relationshipId', '_userId',
                  '_companyName', '_companyPhone', '_isCash', '_BussinessAddress',
                  '_webShopUrl', '_WebShopUrl', '_SchoolName', '_HasBusinessLicense',
                  '_dormitoryPhone', '_incomeFrom', '_schoolName', '_NickName',
                  '_CreationDate', '_CityId', '_DistrictId', '_ProvinceId',
                  '_GraduateDate', '_GraduateSchool', '_IdAddress', '_companySizeId',
                  '_HasPPDaiAccount', '_PhoneType', '_PPDaiAccount', '_SecondEmail',
                  '_SecondMobile', '_nickName', '_HasSbOrGjj', '_Position']

    # group的数量
    userinfo_num = group.shape[0]
    # 修改内容的数量
    userinfo_unique_num = group['UserupdateInfo1'].unique().shape[0]
    # 修改时间的数量
    userinfo_active_day_num = group['UserupdateInfo2'].unique().shape[0]
    # 处理修改时间的最小值
    min_day = parse_date(np.min(group['UserupdateInfo2']))
    # 处理修改时间的最大值
    max_day = parse_date(np.max(group['UserupdateInfo2']))
    # 最小值和最大值相差几天
    gap_day = round((max_day[0] - min_day[0]) / (86400))

    indexes = {
        'userinfo_num': userinfo_num,
        'userinfo_unique_num': userinfo_unique_num,
        'userinfo_active_day_num': userinfo_active_day_num,
        'userinfo_gap_day': gap_day,
        'userinfo_last_day_timestamp': max_day[0]
    }

    for c in op_columns:
        indexes['userinfo' + c + '_num'] = 0

    def sub_aggr(sub_group):
        return sub_group.shape[0]

    sub_group = group.groupby(by=['UserupdateInfo1']).apply(sub_aggr)
    for c in sub_group.index:
        indexes['userinfo' + c + '_num'] = sub_group.loc[c]
    return Series(data=[indexes[c] for c in indexes], index=[c for c in indexes])


train_userinfo_grouped = train_userinfo.groupby(by=['Idx']).apply(userinfo_aggr)
train_userinfo_grouped.head()
train_userinfo_grouped.to_csv('train_userinfo_grouped.csv', header=True, index=True)
train_userinfo_grouped = pd.read_csv('train_userinfo_grouped.csv')
train_userinfo_grouped.head()

print('before merge, train_master shape:{}'.format(train_master_.shape))

# train_master_ = train_master_.merge(train_loginfo_grouped, how='left', left_on='Idx', right_index=True)
# train_master_ = train_master_.merge(train_userinfo_grouped, how='left', left_on='Idx', right_index=True)

train_master_ = train_master_.merge(train_loginfo_grouped, how='left', left_on='Idx', right_on='Idx')
train_master_ = train_master_.merge(train_userinfo_grouped, how='left', left_on='Idx', right_on='Idx')

train_master_.fillna(0, inplace=True)

print('after merge, train_master shape:{}'.format(train_master_.shape))

drop_columns = ['Idx', 'ListingInfo', 'UserInfo_20',  'UserInfo_19', 'UserInfo_8', 'UserInfo_7',
                'UserInfo_4','UserInfo_2',
               'ListingInfo_timestamp', 'loginfo_last_day_timestamp', 'userinfo_last_day_timestamp']
train_master_ = train_master_.drop(drop_columns, axis=1)

dummy_columns = categorical_features.copy()
dummy_columns.extend(['ListingInfo_year', 'ListingInfo_month', 'ListingInfo_day', 'ListingInfo_week',
                      'ListingInfo_isoweekday', 'ListingInfo_month_stage'])
finally_dummy_columns = []

for c in dummy_columns:
    if c not in drop_columns:
        finally_dummy_columns.append(c)

print('before get_dummies train_master_ shape {}'.format(train_master_.shape))
train_master_ = pd.get_dummies(train_master_, columns=finally_dummy_columns)
print('after get_dummies train_master_ shape {}'.format(train_master_.shape))
train_master_.to_csv("master.csv", index=False)

