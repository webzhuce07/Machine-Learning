import tensorflow as tf
print(tf.test.is_gpu_available())
import numpy as np
import matplotlib.pyplot as plt
import math

print(np.percentile(range(1, 101), 75))

import random
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
#用随机数产生一个二维数组。分别是年龄的性别。
df=pd.DataFrame({'Age':np.random.randint(0,70,100),
                'Sex':np.random.choice(['M','F'],100),
                })
#用cut函数对于年龄进行分段分组，用bins来对年龄进行分段，左开右闭
age_groups=pd.cut(df['Age'],bins=[0,18,35,55,70,100])
# print(age_groups)
print(df.groupby(age_groups).count())
print(df.groupby(age_groups).sum())

bins = [0, 20, 40, 60, 100]
labels = [1, 2, 3, 4]
df['AgeRange'] = pd.cut(df['Age'], bins, labels=labels)
print(df.head())

import matplotlib.pyplot as plt  # doctest: +SKIP
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
SVC(random_state=0)
plot_confusion_matrix(clf, X_test, y_test)  # doctest: +SKIP
plt.show()  # doctest: +SKIP