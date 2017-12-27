

"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import sys

import math
import sklearn.metrics as skl_metrics

from datetime import timedelta
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger



#------------------------------------------------------------------------------------#

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    parse_dates=["date"],    
)

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(['id'])

df_T007 = pd.read_csv(
    "../submit/T007_MoreWKs_TrainToAug15.csv", 
).set_index(['id'])

#------------------------------------------------------------------------------------#

df_fcst = pd.concat([df_test,df_T007], axis = 1, join = 'inner')

df_min_date = df_train[['item_nbr','store_nbr','date']].groupby(['item_nbr','store_nbr'])\
    ['date'].min().to_frame('min_date')
df_min_date.reset_index(inplace=True)
df_min_date_recent  = df_min_date.loc[(df_min_date.min_date >= '2017-08-02')]


df_sales_recent = pd.merge(df_min_date_recent, df_train, how='inner', on=['item_nbr','store_nbr'])\
    .groupby(['item_nbr','store_nbr'])['unit_sales'].sum().to_frame('sum_sales')
df_sales_recent.reset_index(inplace=True)

df_sales_recent.sum_sales = df_sales_recent.sum_sales / 14

df_new_items = df_sales_recent[['item_nbr','store_nbr','sum_sales']].groupby(['item_nbr'])\
    ['sum_sales'].mean().to_frame('avg_item_sales')
df_new_items.reset_index(inplace=True)

df_fcst_new = pd.merge(df_fcst,df_new_items,how = 'left',on=['item_nbr'])

### Find index to assign new value 
pos_idx = (df_fcst_new['unit_sales'] == 0) & ( df_fcst_new['avg_item_sales'] > 0),
df_fcst_new_pos = df_fcst_new.loc[pos_idx]

### Assign new value 
#df_fcst_new_pos['unit_sales'] = df_fcst_new_pos['avg_item_sales']
df_fcst_new.loc[pos_idx,'unit_sales'] = df_fcst_new_pos['avg_item_sales']

print('sum of old forecast', df_fcst.unit_sales.sum())
print('sum of new forecast', df_fcst_new.unit_sales.sum())

df_fcst_new[['id','unit_sales']].to_csv('T010_NewSKU_V01.csv.gz', index=False, float_format='%.3f', compression='gzip')




