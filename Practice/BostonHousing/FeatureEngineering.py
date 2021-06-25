import warnings
warnings.filterwarnings('ignore')
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def DataProcessing(df):
    field_cut = {
    'crim' : [0,10,20, 100],
    'zn' : [-1, 5, 18, 20, 40, 80, 86, 100],
    'indus' : [-1, 7, 15, 23, 40],
    'nox' : [0, 0.51, 0.6, 0.7, 0.8, 1],
    'rm' : [0, 4, 5, 6, 7, 8, 9],
    'age' : [0, 60, 80, 100],
    'dis' : [0, 2, 6, 14],
    'rad' : [0, 5, 10, 25],
    'tax' : [0, 200, 400, 500, 800],
    'ptratio' : [0, 14, 20, 23],
    'black' : [0, 100, 350, 450],
    'lstat' : [0, 5, 10, 20, 40]
    }
    cut_df = pd.DataFrame()
    for field in field_cut.keys():
        cut_series = pd.cut(df[field], field_cut[field], right=True)
        onehot_df = pd.get_dummies(cut_series, prefix=field)
        cut_df = pd.concat([cut_df, onehot_df], axis=1)
    new_df = pd.concat([df, cut_df], axis=1)
    return new_df

train.drop(train[train.medv == 50].index)
train_feature = train.iloc[:, 1: -1]
train_feature = DataProcessing(train_feature)


test_feature = train.iloc[:, 1:]
test_feature = DataProcessing(test_feature)

train_feature.to_csv("./train0.csv", index=False)
test_feature.to_csv("./test0.csv", index=False)
