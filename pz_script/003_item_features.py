
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

# import math
# import sklearn.metrics as skl_metrics
# from sklearn.metrics import mean_squared_error

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

##########################################################################

if len(sys.argv) == 1:
    param_1 = "Full Run"
else:
    param_1 = sys.argv[1]
    print("input parameter = ", param_1)

if param_1 == "1s":
    df_train = pd.read_csv(
        '../input/train_2s.csv', usecols=[1, 2, 3, 4, 5],
        dtype={'onpromotion': bool},
        converters={'unit_sales': lambda u: np.log1p(
            float(u)) if float(u) > 0 else 0},
        parse_dates=["date"],
    )

    df_test = pd.read_csv(
        "../input/test_2s.csv", usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]  # , date_parser=parser
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )

else:
    df_train = pd.read_csv(
        '../input/train.csv', usecols=[1, 2, 3, 4, 5],
        dtype={'onpromotion': bool},
        converters={'unit_sales': lambda u: np.log1p(
            float(u)) if float(u) > 0 else 0},
        parse_dates=["date"],
        skiprows=range(1, 23398768)  # 2014-05-06
    )

    df_test = pd.read_csv(
        "../input/test.csv", usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]  # , date_parser=parser
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )

logger.info('Load data successful')

###############################################################################
# Functions


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range
              (dt - timedelta(days=minus), periods=periods, freq=freq)]



def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "item_nbr": df_2017_nbr.item_nbr,
        "date": (t2017), 
        "item_day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "item_mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "item_mean_21_2017": get_timespan(df_2017, t2017, 21, 21).mean(axis=1).values,
        "item_mean_42_2017": get_timespan(df_2017, t2017, 42, 42).mean(axis=1).values,
        "item_mean_91_2017": get_timespan(df_2017, t2017, 91, 91).mean(axis=1).values,
        "item_mean_182_2017": get_timespan(df_2017, t2017, 182, 182).mean(axis=1).values,
        "item_mean_364_2017": get_timespan(df_2017, t2017, 364, 364).mean(axis=1).values,
    })
  
    for i in range(7):
        X['item_dow_4_{}_mean'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
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

df_2017 = df_train.loc[df_train.date >= pd.datetime(2014, 5, 1)]
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


    
df_2017[pd.datetime(2016, 12, 25)] = 0
df_2017[pd.datetime(2015, 12, 25)] = 0
df_2017[pd.datetime(2014, 12, 25)] = 0
if param_1 == "1s":
    df_2017[pd.datetime(2017, 1, 1)] = 0
    df_2017[pd.datetime(2016, 1, 1)] = 0
    df_2017[pd.datetime(2015, 1, 1)] = 0    
    df_2017[pd.datetime(2015, 7, 7)] = 0
#    promo_2017[pd.datetime(2015, 7, 7)] = 0

    
##########################################################################
logger.info('Preparing traing dataset...')

X_l, y_l = [], []

t2016 = date(2016, 8, 3)
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2016 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)

train_week_2017 = 7
if param_1 != "val":
    train_week_2017 = 9

t2017 = date(2017, 5, 31)
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
    train_out.to_pickle('../data/item_train_1s.p')
    val_out.to_pickle('../data/item_val_1s.p')
    X_test.to_pickle('../data/item_test_1s.p')
    
else:
    train_out.to_pickle('../data/item_train.p')
    val_out.to_pickle('../data/item_val.p')
    X_test.to_pickle('../data/item_test.p')
