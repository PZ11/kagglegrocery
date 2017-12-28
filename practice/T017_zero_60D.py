

"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
import sys
import math

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


df_train_maxdate = df_train[['item_nbr','store_nbr','date']].groupby(['item_nbr','store_nbr'])\
    ['date'].max().to_frame('max_date')
df_train_maxdate.reset_index(inplace=True)

del df_train

#----------------------------------------------------------------------------------------#
## Discontinue store/items
df_dis = df_train_maxdate.loc[df_train_maxdate['max_date'] <= '2017-06-16',]

t_prev = pd.read_csv('../submit/T016_mg_ly_wk123_dow.csv')
df_prev = pd.merge(t_prev, df_test,  on=['id'], how = 'inner')


df_sub = pd.merge(df_prev, df_dis,  on=['store_nbr', 'item_nbr'], how = 'inner')
df_sub.unit_sales = 0 

df_sub.to_csv('../input/sub_zero_60d.csv', float_format='%.4f', index=None)

#----------------------------------------------------------------------------------------#

t_new = pd.merge(df_prev, df_sub,  on=['id'], how = 'left')
t_new['unit_sales'] = t_new.unit_sales_y.combine_first(t_new.unit_sales_x)

submission = t_new[['id','unit_sales']]
del t_new

print("Merged Partial, SUM =",  submission.unit_sales.sum())
print("Merged Partial, MEAN =",  submission.unit_sales.mean())

submission.to_csv('../submit/T017_zero_2m_t016.csv', float_format='%.4f', index=None)




