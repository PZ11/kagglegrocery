
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding
more average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
import sys

##########################################################################

prev = pd.read_csv('../submit/T030_train2016.csv')

# Set Froecast to 0.1, T027
prev['unit_sales'] = 1.10 * prev['unit_sales']

submission = prev[['id', 'unit_sales']]


print("Merged  SUM =",  submission.unit_sales.sum())
print("Merged  MEAN =",  submission.unit_sales.mean())

submission.to_csv('../submit/T038_up10P.csv.gz',
                  float_format='%.4f', index=None, compression='gzip')




