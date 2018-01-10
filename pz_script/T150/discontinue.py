

"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import sys

import math
import sklearn.metrics as skl_metrics

from datetime import timedelta
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

df_train = pd.read_csv(
    '../input/train.csv', usecols=[0, 1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    parse_dates=["date"],
)

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
)

#### Remove the items that does not exist in test
#### Memroy Error, ignore this tep
#test_store_item = df_test[['item_nbr','store_nbr','date']].groupby(['item_nbr','store_nbr'])\
#    ['date'].count().to_frame('cnt_date')
#test_store_item.reset_index(inplace=True)
#df_train_new = pd.merge(df_train,test_store_item,how = 'inner',on=['store_nbr','item_nbr'])

### 
df_train_mindate = df_train[['item_nbr','store_nbr','date']].groupby(['item_nbr','store_nbr'])\
    ['date'].min().to_frame('min_date')
    
df_train_mindate.reset_index(inplace=True)
df_train_mindate['date'] = df_train_mindate['min_date']

#### Split Mature and Partial
df_train_p = df_train_mindate.loc[ (df_train_mindate['date'] >'2016-08-10'),]
df_train_m = df_train_mindate.loc[ (df_train_mindate['date'] <='2016-08-10'),]

#### Find store/items 

train_p_store_item = df_train_p[['item_nbr','store_nbr','date']].groupby(['item_nbr','store_nbr'])\
    ['date'].count().to_frame('cnt_date')
train_p_store_item.reset_index(inplace=True)

train_m_store_item = df_train_m[['item_nbr','store_nbr','date']].groupby(['item_nbr','store_nbr'])\
    ['date'].count().to_frame('cnt_date')
train_m_store_item.reset_index(inplace=True)

del df_train_p, df_train_m
####
train_p = pd.merge(df_train,train_p_store_item,how = 'inner',on=['store_nbr','item_nbr']).set_index(['id'])
train_m = pd.merge(df_train,train_m_store_item,how = 'inner',on=['store_nbr','item_nbr']).set_index(['id'])

### delete column
train_m.drop('cnt_date', axis=1, inplace=True)
train_p.drop('cnt_date', axis=1, inplace=True)

print("mature  store/items", train_m.info())
print("partial store/items", train_p.info())

del df_train
train_m.to_csv('../input/train_m.csv')
train_p.to_csv('../input/train_p.csv')

## train_m is failed to write to pickle 
#train_m.to_pickle('../input/train_m.p')
#train_p.to_pickle('../input/train_p.p')

del train_m, train_p

#### Split Test data to mature and partial
test_m = pd.merge(df_test,train_m_store_item,how = 'inner',on=['store_nbr','item_nbr']).set_index(['id'])
print("mature  store/items in test", test_m.info())

test_p= pd.merge(df_test,train_p_store_item,how = 'inner',on=['store_nbr','item_nbr']).set_index(['id'])
print("partial store/items in test", test_p.info())

test_m[['store_nbr', 'item_nbr', 'date', 'onpromotion']].to_csv('../input/test_m.csv')
test_p[['store_nbr', 'item_nbr', 'date', 'onpromotion']].to_csv('../input/test_p.csv')

#test_m[['id','store_nbr', 'item_nbr', 'date', 'onpromotion']].to_pickle('../input/test_m.p')
#test_p[['id','store_nbr', 'item_nbr', 'date', 'onpromotion']].to_pickle('../input/test_p.p')
