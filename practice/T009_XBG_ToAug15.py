
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import sys

import math
import sklearn.metrics as skl_metrics

from datetime import timedelta
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

logger = getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'

DIR = '../logs/'

log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
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

#------------------------------------------------------------------------------------#
if len(sys.argv) == 1:
    param_1 = "Full Run"
else:
    param_1= sys.argv[1] 
    print ("input parameter = ", param_1)

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
	    skiprows=range(1, 66458909)  # 2016-01-01
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

df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
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

#------------------------------------------------------------------------------------------#
# Functions

def NWRMSLE(y, pred, weights=None):
    err2 = skl_metrics.mean_squared_log_error(y, pred, sample_weight=weights)
    return math.sqrt(err2)

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
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
        "mean_150_2017": get_timespan(df_2017, t2017, 150, 150).mean(axis=1).values,
        
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values
    })
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X


   
#------------------------------------------------------------------------------------------#
logger.info('Preparing datasetn...')

t2017 = date(2017, 5, 31)
X_l, y_l = [], []
for i in range(8):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
#X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

#------------------------------------------------------------------------------------------#
logger.info('Training and predicting models...')
logger.info('XGB, Training and predicting models...')


params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 300

val_pred = []
test_pred = []
cate_vars = []
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)

    print("Train a XGBoost model")
    X_xgbtrain = X_train
    y_xgbtrain = y_train[:, i]
    
    dtrain = xgb.DMatrix(X_xgbtrain, y_xgbtrain)

    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
                      early_stopping_rounds=30, verbose_eval=True)

    #importance = bst.get_fscore(fmap='xgb.fmap')
    #print(importance)

    test_pred.append(bst.predict(xgb.DMatrix(X_test)))

del X_train, y_train
#------------------------------------------------------------------------------------------#
# Validate 
#### Need to use expm1 when y is log1p

#------------------------------------------------------------------------------------------#
# Submit
logger.info('Making submission...')

y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('../submit/T009_xgb_moreWK_ToAug15.csv', float_format='%.4f', index=None)

####### PZ, Check overral result
print("SUM =",  submission.unit_sales.sum())
print("MEAN =",  submission.unit_sales.mean())

