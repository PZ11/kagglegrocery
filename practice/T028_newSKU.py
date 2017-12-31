
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding
more average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
import sys

##########################################################################

df_prev = pd.read_csv('../submit/T025_moreLY.csv')
df_sub = pd.read_csv('../input/new__v02_fcst_l9d.csv')

# Set Froecast to 0.1, T027
# df_sub.loc[(df_sub.unit_sales > 0.1),'unit_sales'] = 0.1

t_new = pd.merge(df_prev, df_sub, on=['id'], how='left')
t_new['unit_sales'] = t_new.unit_sales_y.combine_first(t_new.unit_sales_x)

submission = t_new[['id', 'unit_sales']]
del t_new

print("Merged  SUM =",  submission.unit_sales.sum())
print("Merged  MEAN =",  submission.unit_sales.mean())

submission.to_csv('../submit/T028_new_v02_l9d.csv.gz',
                  float_format='%.4f', index=None, compression='gzip')




