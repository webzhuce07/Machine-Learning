
def DataPreprocess(data_train):
    #####异常值处理

    #age异常值处理
    data_train = data_train[data_train['age'] > 0]

    #Num30-59late Num60-89late Num90late异常值处理
    data_train = data_train[data_train['Num30-59late'] < 90]
    data_train = data_train[data_train['Num60-89late'] < 90]
    data_train = data_train[data_train['Num90late'] < 90]

    #Numestate异常值处理
    data_train = data_train[data_train['Numestate'] < 50]


    #####缺失值处理
    # Numdepend缺失值处理
    data_train['Numdepend'] = data_train['Numdepend'].fillna('0')

    # MonthlyIncome缺失值处理
    # 随机森林预测缺失值
    data_Forest = data_train.iloc[:, [5, 1, 2, 3, 4, 6, 7, 8, 9]]
    MonthlyIncome_isnull = data_Forest.loc[data_train['MonthlyIncome'].isnull(), :]
    MonthlyIncome_notnull = data_Forest.loc[data_train['MonthlyIncome'].notnull(), :]

    from sklearn.ensemble import RandomForestRegressor
    X = MonthlyIncome_notnull.iloc[:, 1:].values
    y = MonthlyIncome_notnull.iloc[:, 0].values
    regr = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=200, n_jobs=-1)
    regr.fit(X, y)

    print(MonthlyIncome_isnull.iloc[:, 1:].info())
    MonthlyIncome_fillvalue = regr.predict(MonthlyIncome_isnull.iloc[:, 1:].values)

    # 填充MonthlyIncome缺失值
    data_train.loc[data_train['MonthlyIncome'].isnull(), 'MonthlyIncome'] = MonthlyIncome_fillvalue

    # 衍生变量
    data_train['AllNumlate'] = data_train['Num30-59late'] + data_train['Num60-89late'] + data_train['Num90late']
    data_train['Monthlypayment'] = data_train['DebtRatio'] * data_train['MonthlyIncome']
    data_train['Withdepend'] = data_train['Numdepend']

    # 数据类型转换
    data_train['Numdepend'] = data_train['Numdepend'].astype('int64')
    data_train['Withdepend'] = data_train['Withdepend'].astype('int64')
    data_train['MonthlyIncome'] = data_train['MonthlyIncome'].astype('int64')
    data_train['Monthlypayment'] = data_train['Monthlypayment'].astype('int64')

    # Revol分箱
    data_train.loc[(data_train['Revol'] < 1), 'Revol'] = 0
    data_train.loc[(data_train['Revol'] > 1) & (data_train['Revol'] <= 20), 'Revol'] = 1
    data_train.loc[(data_train['Revol'] > 20), 'Revol'] = 0  # 根据前文EDA分析，将大于20的数据与0-1的数据合并

    # DebtRatio分箱
    data_train.loc[(data_train['DebtRatio'] < 1), 'DebtRatio'] = 0
    data_train.loc[(data_train['DebtRatio'] > 1) & (data_train['DebtRatio'] < 2), 'DebtRatio'] = 1
    data_train.loc[(data_train['DebtRatio'] >= 2), 'DebtRatio'] = 0

    # Num30-59late/Num60-89late/Num90late/Numestate/Numdepend
    data_train.loc[(data_train['Num30-59late'] >= 8), 'Num30-59late'] = 8
    data_train.loc[(data_train['Num60-89late'] >= 7), 'Num60-89late'] = 7
    data_train.loc[(data_train['Num90late'] >= 10), 'Num90late'] = 10
    data_train.loc[(data_train['Numestate'] >= 8), 'Numestate'] = 8
    data_train.loc[(data_train['Numdepend'] >= 7), 'Numdepend'] = 7

    # AllNumlate分箱
    data_train.loc[(data_train['AllNumlate'] > 1), 'AllNumlate'] = 1  # 分为逾期和未逾期两种情况

    # Withdepend分箱
    data_train.loc[(data_train['Withdepend'] > 1), 'Withdepend'] = 1  # 分为独生子女和非独生子女

    return data_train
