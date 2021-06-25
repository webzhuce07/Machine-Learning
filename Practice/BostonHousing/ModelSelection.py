import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd

train = pd.read_csv('train0.csv')
xgb_model = XGBRegressor(nthread=7)
cv_split = ShuffleSplit(n_splits=6, train_size=0.7, test_size=0.2)
grid_params = dict(
    max_depth=[4, 5, 6, 7],
    learning_rate=np.linspace(0.03, 0.3, 10),
    n_estimators=[100, 200]
)
grid = GridSearchCV(xgb_model, grid_params, cv=cv_split, scoring='neg_mean_squared_error')
X = train.iloc[:, 1: -1].values
y = train.iloc[:, -1].values
grid.fit(X, y)
print(grid.best_params_)