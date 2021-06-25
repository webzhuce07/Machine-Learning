import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from xgboost import XGBRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.iloc[:, 1: -1].values
y = train.iloc[:, -1].values
model = XGBRegressor(learning_rate=0.03, max_depth=4, n_estimators=100)
model.fit(X, y)
test_feature = test.iloc[:, 1:].values
prediction = model.predict(test_feature)
result = pd.DataFrame({'ID': test.ID, 'medv': prediction})
result.to_csv("result.csv", index=False)