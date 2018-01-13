
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding
more average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
import sys

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
)

##########################################################################

df_sub = pd.read_csv('../input/output_ZeroForecastInLess30Ds.csv')
df_sub['unit_sales'] = 0
df_sub = pd.merge(df_sub, df_test, on=['store_nbr', 'item_nbr'],  how = 'inner')

print(df_sub.shape)
print(df_sub.info())

# Load previous forecast 
df_prev = pd.read_csv('../submit/T300_V3.csv')


t_new = pd.merge(df_prev, df_sub[['id', 'unit_sales']], on=['id'], how='left')
t_new['unit_sales'] = t_new.unit_sales_y.combine_first(t_new.unit_sales_x)

submission = t_new[['id', 'unit_sales']]
del t_new

print("Merged  SUM =",  submission.unit_sales.sum())
print("Merged  MEAN =",  submission.unit_sales.mean())

submission.to_csv('../submit/T300_V4.csv.gz',
                  float_format='%.4f', index=None, compression='gzip')




