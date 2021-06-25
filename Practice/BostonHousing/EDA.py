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

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#了解缺失值情况和每列的数据类型
print(train.info())
print(test.info())
print(train.describe())
print(test.describe())
#没有缺失值，数据类型是float和int

df_y = pd.DataFrame({'y_':train.SalePrice,'y_log1p':np.log1p(train.SalePrice)})
df_y.hist()