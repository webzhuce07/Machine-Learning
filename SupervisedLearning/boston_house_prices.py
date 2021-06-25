from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from numpy import shape

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target
# print(shape(data_X))
# print(shape(data_y))
# print(data_X[:2, :])
# print(data_y[:2])

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)
# print(shape(X_train))
# print shape(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
# print (model.coef_)
# print (model.intercept_)

y_pred = model.predict(X_test)
from sklearn import metrics

print
"MSE:", metrics.mean_squared_error(y_test, y_pred)

predicted = cross_val_predict(model, data_X, data_y, cv=10)
print
"MSE:", metrics.mean_squared_error(data_y, predicted)

plt.scatter(data_y, predicted, color='y', marker='o')
plt.scatter(data_y, data_y, color='g', marker='+')
plt.show()