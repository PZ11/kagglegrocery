
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

        "sum_ty_p2_7d": get_timespan(df_2017, t2017, 14 , 7).mean(axis=1).values,
        "sum_ty_p2_14d": get_timespan(df_2017, t2017, 28 , 14).mean(axis=1).values,
        "sum_ty_p2_30d": get_timespan(df_2017, t2017, 60 , 30).mean(axis=1).values,
        "sum_ty_p2_60d": get_timespan(df_2017, t2017, 120 , 60).mean(axis=1).values,

        "sum_ly_p7d": get_timespan(df_2017, t2017, 371 , 7).mean(axis=1).values,        
        "sum_ly_p14d": get_timespan(df_2017, t2017, 378 , 14).mean(axis=1).values,
        "sum_ly_p30d": get_timespan(df_2017, t2017, 394 , 30).mean(axis=1).values,
        "sum_ly_p60d": get_timespan(df_2017, t2017, 424 , 60).mean(axis=1).values,


        "family": df_2017_nbr.family,
        "store_nbr": df_2017_nbr.store_nbr,
        "date": (t2017), 
        "s_f_day_1": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "s_f_mean_7": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "s_f_mean_14": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "s_f_mean_30": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
        "s_f_mean_60": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "s_f_mean_91": get_timespan(df_2017, t2017, 91, 91).mean(axis=1).values,
        "s_f_mean_182": get_timespan(df_2017, t2017, 182, 182).mean(axis=1).values,
        "s_f_mean_364": get_timespan(df_2017, t2017, 364, 364).mean(axis=1).values,
    })
  
    for i in range(7):
        X['s_f_dow_4_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['s_f_dow_13_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 91-i, 13, freq='7D').mean(axis=1).values
        X['s_f_dow_26_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 182-i, 26, freq='7D').mean(axis=1).values
        X['s_f_dow_52_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 364-i, 52, freq='7D').mean(axis=1).values        


    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X


def calc_ratio(df):
    
    df['____s_f_ratio_tyly_7d'] = df['s_f_mean_7'] / df['sum_ly_p7d']
    df['____s_f_ratio_tyly_14d'] = df['s_f_mean_14'] / df['sum_ly_p14d']
    df['____s_f_ratio_tyly_30d'] = df['s_f_mean_30'] / df['sum_ly_p30d']
    df['____s_f_ratio_tyly_60d'] = df['s_f_mean_60'] / df['sum_ly_p60d']

    df['____s_f_ratio_ty_p_7d'] = df['s_f_mean_7'] / df['sum_ty_p2_7d']
    df['____s_f_ratio_ty_p_14d'] = df['s_f_mean_14'] / df['sum_ty_p2_14d']
    df['____s_f_ratio_ty_p_30d'] = df['s_f_mean_30'] / df['sum_ty_p2_30d']
    df['____s_f_ratio_ty_p_60d'] = df['s_f_mean_60'] / df['sum_ty_p2_60d']


    del df['sum_ty_p2_7d'], df['sum_ty_p2_14d'], df['sum_ty_p2_30d'], df['sum_ty_p2_60d']
    del df['sum_ly_p14d'], df['sum_ly_p30d'], df['sum_ly_p60d'], df['sum_ly_p7d']

    return df

###############################################################################
# Aggregate to s-f(store-family) level

df_train_store_items = pd.merge(df_train, items, on =['item_nbr'], how = 'inner')

df_train_store_family = df_train_store_items[['family', 'date', 'store_nbr', 'unit_sales', 'item_nbr']]\
                        .groupby(['family','store_nbr','date'])\
                        .agg({'unit_sales': 'sum', 'item_nbr':'count'}).reset_index()
df_train_store_family["item_avg_sales"] = df_train_store_family["unit_sales"] / df_train_store_family["item_nbr"]

df_2017 = df_train_store_family.set_index(
    ["family", "store_nbr", "date"])[["item_avg_sales"]].unstack(
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

# Always load 9 weeks of data. if val, 2 weeks will be removed in 100_model. 
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

X_train = calc_ratio(X_train).fillna(0)
X_val = calc_ratio(X_val).fillna(0)
X_test = calc_ratio(X_test).fillna(0)

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
    train_out.to_pickle('../data/storefamily_train_1s.p')
    val_out.to_pickle('../data/storefamily_val_1s.p')
    X_test.to_pickle('../data/storefamily_test_1s.p')
    
else:
    train_out.to_pickle('../data/storefamily_train.p')
    val_out.to_pickle('../data/storefamily_val.p')
    X_test.to_pickle('../data/storefamily_test.p')
