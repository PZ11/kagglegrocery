
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

from load_data import load_input_data, add_missing_days, add_missing_days_nopromo

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

items = pd.read_csv("../input/items.csv",)

t2014 = date(2014, 8, 6)
t2015 = date(2015, 8, 5)
t2016 = date(2016, 8, 3)
t2017 = date(2017, 5, 31)
train_week_2017 = 9

logger.info('Load data successful')

###############################################################################
# Functions


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range
              (dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "class": df_2017_nbr['class'],
        "date": (t2017),       
        "____class_day_01_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "____class_day_03_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "____class_mean_07_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "____class_mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "____class_mean_21_2017": get_timespan(df_2017, t2017, 21, 21).mean(axis=1).values,
        "____class_mean_42_2017": get_timespan(df_2017, t2017, 42, 42).mean(axis=1).values,
        "____class_mean_91_2017": get_timespan(df_2017, t2017, 91, 91).mean(axis=1).values,
        "____class_mean_182_2017": get_timespan(df_2017, t2017, 182, 182).mean(axis=1).values,
        "____class_mean_364_2017": get_timespan(df_2017, t2017, 364, 364).mean(axis=1).values,
    })
  
    for i in range(7):
        X['__class_dow_01_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 7-i,1).values.ravel()
        X['__class_dow_03_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 21-i, 3, freq='7D').mean(axis=1).values
        X['__class_dow_06_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 42-i, 6, freq='7D').mean(axis=1).values        
        X['__class_dow_13_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 91-i, 13, freq='7D').mean(axis=1).values
        X['__class_dow_26_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 182-i, 26, freq='7D').mean(axis=1).values
        X['__class_dow_52_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 364-i, 52, freq='7D').mean(axis=1).values     

    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

###############################################################################
# Aggregate to class level

df_train_item_class = pd.merge(df_train, items, on =['item_nbr'], how = 'inner')

df_train_class = df_train_item_class[['class', 'date', 'unit_sales', 'item_nbr']]\
                        .groupby(['class','date'])\
                        .agg({'unit_sales': 'sum', 'item_nbr':'count'}).reset_index()
df_train_class["class_avg_sales"] = df_train_class["unit_sales"] / df_train_class["item_nbr"]

df_2017 = df_train_class.set_index(
    ["class", "date"])[["class_avg_sales"]].unstack(
        level=-1).fillna(0)

df_2017.columns = df_2017.columns.get_level_values(1)

df_2017_nbr = pd.DataFrame(df_2017.copy())
df_2017_nbr.reset_index(inplace = True)

    
df_2017 = add_missing_days_nopromo(df_2017, param_1)
    
##########################################################################
logger.info('Preparing traing dataset...')

X_l, y_l = [], []

# Add train data on Aug 2014 and Aug 2015
logger.info('Preparing 2014 training dataset...')
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2014 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)

logger.info('Preparing 2015 training dataset...')
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2015 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)

logger.info('Preparing 2016 training dataset...')
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2016 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)

logger.info('Preparing 2017 training dataset...')
for i in range(train_week_2017):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)


X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l

delta = timedelta(0)

X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

##########################################################################
logger.info('Save Store Item Features ...')

y_columns = ["day" + str(i) for i in range(1, 17)]

df_y_train = pd.DataFrame(data = y_train, columns = y_columns)
X_train.reset_index(inplace = True)
X_train.reindex(index = df_y_train.index)
#train_out = pd.concat([X_train, df_y_train], axis = 1) 
train_out = X_train

df_y_val = pd.DataFrame(data = y_val, columns = y_columns)
X_val.reset_index(inplace = True)
X_val.reindex(index = df_y_val.index)
#val_out = pd.concat([X_val, df_y_val], axis = 1)
val_out = X_val

##########################################################################
# output


if param_1 == "1s":
    train_out.to_pickle('../data/class_train_1s.p')
    val_out.to_pickle('../data/class_val_1s.p')
    X_test.to_pickle('../data/class_test_1s.p')
    
else:
    train_out.to_pickle('../data/class_train.p')
    val_out.to_pickle('../data/class_val.p')
    X_test.to_pickle('../data/class_test.p')
