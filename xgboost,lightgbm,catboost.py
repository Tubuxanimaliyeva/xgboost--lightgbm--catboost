# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:56:28 2022

@author: User
"""

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb



y.value_counts()


boston_dataset = datasets.load_boston()


boston = pd.DataFrame(boston_dataset.data)
boston.columns = boston_dataset.feature_names
boston['HOUSE_PRICE'] = boston_dataset.target


x = boston.iloc[:, :-1]
y = boston[['HOUSE_PRICE']]


x_train, x_test, y_train, y_test = \
train_test_split(x, y, test_size = 0.2, random_state = 1)


sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_train))
x_test = pd.DataFrame(sc.transform(x_test))

le = LabelEncoder()
y_train = le.fit_transform(y_train)
______________________________________________________________________________________________________________________________
xgb = xgb.XGBRFClassifier()
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)

mse_xgb = ((y_test-xgb_pred.reshape(102, 1))**2).mean()
__________________________________________________________________________________________________________________________________
lgb = lgbm.LGBMClassifier()
lgb.fit(x_train, y_train)
lgb_pred = lgb.predict(x_test)

mse_lgb = ((y_test-lgb_pred.reshape(102, 1))**2).mean()
________________________________________________________________________________________________________________________-

cb = cb.CatBoostClassifier()
cb.fit(x_train, y_train)
cb_pred = cb.predict(x_test)

mse_cb = ((y_test-cb_pred.reshape(102, 1))**2).mean()
 
 
 





