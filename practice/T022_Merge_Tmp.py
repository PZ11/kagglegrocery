
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding
more average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys

##########################################################################

df_prev = pd.read_csv('../submit/T022_tmp.csv')
df_sub = pd.read_csv('../input/sub_zero_30d.csv')

t_new = pd.merge(df_prev, df_sub, on=['id'], how='left')
t_new['unit_sales'] = t_new.unit_sales_y.combine_first(t_new.unit_sales_x)

submission = t_new[['id', 'unit_sales']]
del t_new

print("Merged  SUM =",  submission.unit_sales.sum())
print("Merged  MEAN =",  submission.unit_sales.mean())

submission.to_csv('../submit/T022_t9wk_moreLY.csv.gz',
                  float_format='%.4f', index=None, compression='gzip')
