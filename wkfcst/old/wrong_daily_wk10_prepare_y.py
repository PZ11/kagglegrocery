
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding
more average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import math
import gc
import sklearn.metrics as skl_metrics

from load_data import load_input_data, add_missing_days

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

logger = getLogger(__name__)

pd.options.mode.chained_assignment = None  # default='warn'

DIR = '../logs/'

log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s]\
    [%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(DIR + 'train.py.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)

logger.info('start')


if len(sys.argv) == 1:
    param_1 = "Full Run"
else:
    param_1 = sys.argv[1]
    print("input parameter = ", param_1)

df_train, df_test = load_input_data(param_1)

items = pd.read_csv("../input/items.csv",).set_index("item_nbr")

t2014 = date(2014, 8, 6)
t2015 = date(2015, 8, 5)
t2016 = date(2016, 8, 3)
t2017 = date(2017, 5, 31)
train_week_2017 = 9

logger.info('Load data successful')

###############################################################################
# Functions

def prepare_dataset(t2017, is_train=True):
    y = df_2017[ pd.date_range(t2017, periods=16) ].values
    return y


###############################################################################

df_2017 = df_train.loc[df_train.date >= pd.datetime(2013, 5, 1)]

del df_train
gc.collect()


df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

    
##########################################################################
logger.info('Preparing traing dataset...')

y_l = []

# Add train data on Aug 2014 and Aug 2015
logger.info('Preparing 2014 training dataset...')
for i in range(4):
    delta = timedelta(days=7 * i)
    y_tmp = prepare_dataset(t2014 + delta)
    y_l.append(y_tmp)

logger.info('Preparing 2015 training dataset...')
for i in range(4):
    delta = timedelta(days=7 * i)
    y_tmp = prepare_dataset(t2015 + delta)
    y_l.append(y_tmp)


logger.info('Preparing 2016 training dataset...')
for i in range(4):
    delta = timedelta(days=7 * i)
    y_tmp = prepare_dataset(t2016 + delta  )
    y_l.append(y_tmp)


# Always load 9 weeks of data. if val, 2 weeks will be removed in 100_model. 
logger.info('Preparing 2017 training dataset...')
for i in range(train_week_2017):
    delta = timedelta(days=7 * i)
    y_tmp = prepare_dataset( t2017 + delta )
    y_l.append(y_tmp)

y_train = np.concatenate(y_l, axis=0)
del y_l

print("total Train set: ", y_train.shape)

delta = timedelta(0)
y_val = prepare_dataset(date(2017, 7, 26))

##########################################################################
logger.info('Save Store Item Features ...')

y_columns = ["day" + str(i) for i in range(1, 17)]

df_y_train = pd.DataFrame(data = y_train, columns = y_columns)

train_shape = df_y_train.shape
logger.info(train_shape)
train_info = df_y_train.info()
logger.info(train_info)

df_y_val = pd.DataFrame(data = y_val, columns = y_columns)


##########################################################################
# output

if param_1 == "1s":
    df_y_train.to_pickle('../wkdata/y_train_1s.p')
    df_y_val.to_pickle('../wkdata/y_val_1s.p')
    
else:
    df_y_train.to_pickle('../wkdata/y_train.p')
    df_y_val.to_pickle('../wkdata/y_val.p')

