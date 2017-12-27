
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import sys

import math
import sklearn.metrics as skl_metrics

from datetime import timedelta
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

#------------------------------------------------------------------------------------------#
#### Merge partial result with T007

p_sub = pd.read_csv('../submit/T012_p.csv')
t_pref = pd.read_csv('../submit/T007_MoreWKs_TrainToAug15.csv')

t_new = pd.merge(t_pref, p_sub,  on=['id'], how = 'left')
t_new['unit_sales'] = t_new.unit_sales_y.combine_first(t_new.unit_sales_x)

submit_s1 = t_new[['id','unit_sales']]
del t_new

print("Merged Partial, SUM =",  submit_s1.unit_sales.sum())
print("Merged Partial, MEAN =",  submit_s1.unit_sales.mean())


#------------------------------------------------------------------------------------------#
#### Merge mature result with T007

m_sub = pd.read_csv('../submit/T012_m.csv')

t_new = pd.merge(submit_s1, m_sub,  on=['id'], how = 'left')
t_new['unit_sales'] = t_new.unit_sales_y.combine_first(t_new.unit_sales_x)

submission = t_new[['id','unit_sales']]
del t_new

print("Merged mature, SUM =",  submission.unit_sales.sum())
print("Merged Mature. MEAN =",  submission.unit_sales.mean())

submission.to_csv('../submit/T012_merged_p_m.csv', float_format='%.4f', index=None)
