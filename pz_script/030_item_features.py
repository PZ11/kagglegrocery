
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

items = pd.read_csv("../input/items.csv",).set_index("item_nbr")

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



def calc_ratio(df):
    
    df['____ratio_tyly_7d'] = df['item_mean_07'] / df['sum_ly_p7d']
    df['____ratio_tyly_21d'] = df['item_mean_21'] / df['sum_ly_p21d']

    df['____ratio_ty_p_7d'] = df['item_mean_07'] / df['sum_ty_p2_7d']
    df['____ratio_ty_p_21d'] = df['item_mean_21'] / df['sum_ty_p2_21d']
    df['____ratio_ty_p_42d'] = df['item_mean_42'] / df['sum_ty_p2_42d']

    del df['sum_ty_p2_7d'], df['sum_ty_p2_21d'], df['sum_ty_p2_42d']
    del df['sum_ly_p7d'], df['sum_ly_p21d']

    return df


def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({

        "item_nbr": df_2017_nbr.item_nbr,
        "date": (t2017),       
        "item_day_01": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "item_mean_07": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "item_mean_21": get_timespan(df_2017, t2017, 21, 21).mean(axis=1).values,
        "item_mean_42": get_timespan(df_2017, t2017, 42, 42).mean(axis=1).values,
        "item_mean_91": get_timespan(df_2017, t2017, 91, 91).mean(axis=1).values,
        "item_mean_182": get_timespan(df_2017, t2017, 182, 182).mean(axis=1).values,
        "item_mean_364": get_timespan(df_2017, t2017, 364, 364).mean(axis=1).values,

    })
  
    for i in range(7):
        X['item_dow_04_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['item_dow_13_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 91-i, 13, freq='7D').mean(axis=1).values
        X['item_dow_26_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 182-i, 26, freq='7D').mean(axis=1).values
        X['item_dow_52_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 364-i, 52, freq='7D').mean(axis=1).values     

    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

###############################################################################

df_2017 = df_train.loc[df_train.date >= pd.datetime(2013, 5, 1)]
del df_train


# Aggregate to item level
df_train_item = df_2017[['item_nbr','date', 'store_nbr', 'unit_sales']].groupby(['item_nbr','date'])\
    .agg({'unit_sales': 'sum', 'store_nbr':'count'}).reset_index()

df_train_item["item_avg_sales"] = df_train_item["unit_sales"] / df_train_item["store_nbr"]

df_2017 = df_train_item.set_index(
    ["item_nbr", "date"])[["item_avg_sales"]].unstack(
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
logger.info('Save Item Features ...')

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

# Comment on T290
#X_train = calc_ratio(X_train).fillna(0)
#X_val = calc_ratio(X_val).fillna(0)
#X_test = calc_ratio(X_test).fillna(0)

##########################################################################
# output


if param_1 == "1s":
    train_out.to_pickle('../data/item_train_1s.p')
    val_out.to_pickle('../data/item_val_1s.p')
    X_test.to_pickle('../data/item_test_1s.p')
    
else:
    train_out.to_pickle('../data/item_train.p')
    val_out.to_pickle('../data/item_val.p')
    X_test.to_pickle('../data/item_test.p')
