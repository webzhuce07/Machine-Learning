import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_master = pd.read_csv("./Training Set/PPD_Training_Master_GBK_3_1_Training_Set.csv", encoding='gbk')
train_log = pd.read_csv("./Training Set/PPD_LogInfo_3_1_Training_Set.csv", encoding='gbk')
train_update = pd.read_csv("./Training Set/PPD_Userupdate_Info_3_1_Training_Set.csv", encoding='gbk')

test_master = pd.read_csv("./Test Set/PPD_Master_GBK_2_Test_Set.csv", encoding='gb18030')
test_log = pd.read_csv("./Test Set/PPD_LogInfo_2_Test_Set.csv", encoding='gbk')
test_update = pd.read_csv("./Test Set/PPD_Userupdate_Info_2_Test_Set.csv", encoding='gbk')

#了解缺失值情况和每列的数据类型
print(train_master.info())
print(train_log.info())
print(train_update.info())
print(test_master.info())
print(test_log.info())
print(test_update.info())

# UserInfo_8清洗处理，处理后非重复项计数减小到400
train_master['UserInfo_8'] = [s[:-1] if s.find('市') > 0 else s[:] for s in train_master.UserInfo_8]
train_update['UserupdateInfo1'] = train_update.UserupdateInfo1.map(lambda x : x.lower())
train_master['ListingInfo'] = pd.to_datetime(train_master.ListingInfo)

import datetime as dt
train_master['month'] = train_master['ListingInfo'].map(lambda x: x.strftime('%Y-%m'))
train_master['month'] = train_master['ListingInfo'].dt.strftime('%Y-%m')
# print(train_master['ListingInfo'])
# print(pd.Index(train_master['ListingInfo']).dayofweek)
# print(pd.Index(train_master['ListingInfo']).year)
# print(pd.Index(train_master['ListingInfo']).hour)
print(pd.Index(train_master['ListingInfo']).day_name)
train_master['DayName'] = train_master['ListingInfo'].apply(lambda d: d.day_name)