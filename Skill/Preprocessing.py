import numpy as np
from sklearn import preprocessing

print("StandardScaler:")
X_train = np.array([[1., -1.,  2.],
                    [2.,  0.,  0.],
                    [0.,  1., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler)
print(scaler.mean_)
print(scaler.scale_)
print(scaler.transform(X_train))

print("MinMaxScaler:")
X_train = np.array([[1., -1.,  2.],
                    [2.,  0.,  0.],
                    [0.,  1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)

print("OrdinalEncoder:")
enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
print(enc.transform([['female', 'from US', 'uses Safari']]))

print("OneHotEncoder:")
enc = preprocessing.OneHotEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
print(enc.transform([['female', 'from US', 'uses Safari'],
                ['male', 'from Europe', 'uses Safari']]).toarray())
