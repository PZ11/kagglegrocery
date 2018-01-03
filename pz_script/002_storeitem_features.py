
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
        '../input/train_1s.csv', usecols=[1, 2, 3, 4, 5],
        dtype={'onpromotion': bool},
        converters={'unit_sales': lambda u: np.log1p(
            float(u)) if float(u) > 0 else 0},
        parse_dates=["date"],
    )

    df_test = pd.read_csv(
        "../input/test_1s.csv", usecols=[0, 1, 2, 3, 4],
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


items = pd.read_csv(
    "../input/items.csv",
).set_index("item_nbr")

logger.info('Load data successful')

###############################################################################
# Functions


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range
              (dt - timedelta(days=minus), periods=periods, freq=freq)]


def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "item_nbr": df_2017_nbr.item_nbr,
        "store_nbr": df_2017_nbr.store_nbr,
        "date": (t2017), 
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,

        "mean_21_2017": get_timespan(df_2017, t2017, 21, 21).mean(axis=1).values,
        "mean_42_2017": get_timespan(df_2017, t2017, 42, 42).mean(axis=1).values,
        "mean_91_2017": get_timespan(df_2017, t2017, 91, 91).mean(axis=1).values,
        "mean_182_2017": get_timespan(df_2017, t2017, 182, 182).mean(axis=1).values,
        "mean_364_2017": get_timespan(df_2017, t2017, 364, 364).mean(axis=1).values,

       #"mean_ly_wk1_2017": get_timespan(df_2017, t2017, 364, 7).mean(axis=1).values,
       #"mean_ly_wk2_2017": get_timespan(df_2017, t2017, 364, 14).mean(axis=1).values,        
        "mean_ly_n16d_2017": get_timespan(df_2017, t2017, 364, 16).mean(axis=1).values,

        "mean_ly_7_2017":  get_timespan(df_2017, t2017, 371, 7 ).mean(axis=1).values,
        "mean_ly_14_2017": get_timespan(df_2017, t2017, 378, 14).mean(axis=1).values,
        "mean_ly_30_2017": get_timespan(df_2017, t2017, 394, 30).mean(axis=1).values,
        "mean_ly_21_2017": get_timespan(df_2017, t2017, 385, 21).mean(axis=1).values,

        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values
    })

    for i in range(16):
        X['ly_1d_d{}'.format(i)] = get_timespan(df_2017, t2017, 364-i, 1).values.ravel()
        X['l2y_1d_d{}'.format(i)] = get_timespan(df_2017, t2017, 728-i, 1).values.ravel()        
        
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
        X['mean_52_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 364-i, 52, freq='7D').mean(axis=1).values        
        X['mean_ly3w_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 364-i, 3, freq='7D').mean(axis=1).values
        X['mean_ly8w_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 392-i, 7, freq='7D').mean(axis=1).values

    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)

    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X


###############################################################################

df_2017 = df_train.loc[df_train.date >= pd.datetime(2014, 5, 1)]
del df_train

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))

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
train_out = pd.concat([X_train, df_y_train], axis = 1)

df_y_val = pd.DataFrame(data = y_val, columns = y_columns)
X_val.reset_index(inplace = True)
X_val.reindex(index = df_y_val.index)
val_out = pd.concat([X_val, df_y_val], axis = 1)

##########################################################################
# output


if param_1 == "1s":
    train_out.to_pickle('../data/storeitem_train_1s.p')
    val_out.to_pickle('../data/storeitem_val_1s.p')
    X_test.to_pickle('../data/storeitem_test_1s.p')
    
else:
    train_out.to_pickle('../data/storeitem_train.p')
    val_out.to_pickle('../data/storeitem_val.p')
    X_test.to_pickle('../data/storeitem_test.p')
