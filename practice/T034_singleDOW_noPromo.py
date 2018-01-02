
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
        skiprows=range(1, 45811211)  # 2016-01-01
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


def NWRMSLE(y, pred, weights=None):
    err2 = skl_metrics.mean_squared_log_error(y, pred, sample_weight=weights)
    return math.sqrt(err2)


def eval_test(test_e):

    test_e['weights'] = 1
    test_e.loc[(test_e.perishable == 1), ('weights')] = 1.25

    result = NWRMSLE(test_e.unit_sales.astype(np.float64), test_e.pred_sales.
                     astype(np.float64), test_e.weights)

    print("Eval All, Number of rows in test is", test_e.shape[0])
    print("Eval all, Forecast Period From:", min(test_e.date),
          " To: ", max(test_e.date))

    # check result on first 6 days.
    test_p1 = test_e.loc[(test_e.date < '2017-08-01'), ]
    result_p1 = NWRMSLE(test_p1.unit_sales.astype(np.float32),
                        test_p1.pred_sales.astype(np.float32), test_p1.weights)

    print("Eval P1, Number of rows in test is", test_p1.shape[0])
    print("Eval P1, Forecast Period From:", min(test_p1.date),
          " To: ", max(test_p1.date))

    # check result on last 10 days.
    test_p2 = test_e.loc[(test_e.date >= '2017-08-01'), ]
    result_p2 = NWRMSLE(test_p2.unit_sales.astype(np.float32),
                        test_p2.pred_sales.astype(np.float32), test_p2.weights)

    print("Eval P2, Number of rows in test is", test_p2.shape[0])
    print("Eval P2, Forecast Period From:", min(test_p2.date),
          " To: ", max(test_p2.date))

    print("Eval All Weighted NWRMSLE = ", result)
    print("Eval P1  Weighted NWRMSLE = ", result_p1)
    print("Eval P2  Weighted NWRMSLE = ", result_p2)

    test_e['error'] = abs(test_e.pred_sales - test_e.unit_sales)
    print("Bias =", (test_e.pred_sales.sum() - test_e.unit_sales.sum())
          / test_e.unit_sales.sum())
    print("WMAPE =", abs(test_e.error.sum() - test_e.unit_sales.sum())
          / test_e.unit_sales.sum())


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range
              (dt - timedelta(days=minus), periods=periods, freq=freq)]


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
        "mean_182_2017": get_timespan(df_2017, t2017, 182, 182).mean(axis=1).values,
        "mean_364_2017": get_timespan(df_2017, t2017, 364, 364).mean(axis=1).values,

       #"mean_ly_wk1_2017": get_timespan(df_2017, t2017, 364, 7).mean(axis=1).values,
       #"mean_ly_wk2_2017": get_timespan(df_2017, t2017, 364, 14).mean(axis=1).values,        
        "mean_ly_n16d_2017": get_timespan(df_2017, t2017, 364, 16).mean(axis=1).values,

        "mean_ly_7_2017":  get_timespan(df_2017, t2017, 371, 7 ).mean(axis=1).values,
        "mean_ly_14_2017": get_timespan(df_2017, t2017, 378, 14).mean(axis=1).values,
        "mean_ly_30_2017": get_timespan(df_2017, t2017, 394, 30).mean(axis=1).values,
        "mean_ly_21_2017": get_timespan(df_2017, t2017, 385, 21).mean(axis=1).values,

#        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
#        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
#        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values
    })

    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
        X['mean_52_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 364-i, 52, freq='7D').mean(axis=1).values        
        X['mean_ly3w_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 364-i, 3, freq='7D').mean(axis=1).values
        X['mean_ly8w_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 392-i, 7, freq='7D').mean(axis=1).values

#    for i in range(16):
#        X["promo_{}".format(i)] = promo_2017[
#            t2017 + timedelta(days=i)].values.astype(np.uint8)

    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X


###############################################################################

df_2017 = df_train.loc[df_train.date >= pd.datetime(2015, 5, 1)]
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


df_2017[pd.datetime(2016, 12, 25)] = 0
df_2017[pd.datetime(2015, 12, 25)] = 0
if param_1 == "1s":
    df_2017[pd.datetime(2017, 1, 1)] = 0
    df_2017[pd.datetime(2016, 1, 1)] = 0
    df_2017[pd.datetime(2015, 7, 7)] = 0

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


X_train_allF = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l

X_val_allF, y_val = prepare_dataset(date(2017, 7, 26))
X_test_allF = prepare_dataset(date(2017, 8, 16), is_train=False)

##########################################################################
logger.info('Training and predicting models...')

params = {
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'num_threads': 4
}

MAX_ROUNDS = 500
val_pred = []
test_pred = []
cate_vars = []
features_all = X_train_allF.columns.tolist()

for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    features_t = features_all.copy()

#    for j in range(16):
#        if j != i:
#            features_t.remove("promo_{}".format(j))

    for j in range(7):
        if j != i%7:
            features_t.remove('mean_4_dow{}_2017'.format(j))
            features_t.remove('mean_20_dow{}_2017'.format(j))
            features_t.remove('mean_52_dow{}_2017'.format(j))
            features_t.remove('mean_ly3w_dow{}_2017'.format(j))
            features_t.remove('mean_ly8w_dow{}_2017'.format(j))

    X_train = X_train_allF[features_t]
    X_val = X_val_allF[features_t]
    X_test = X_test_allF[features_t]

    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * (train_week_2017 + 4)) * 0.25 + 1
    )

    if (param_1 == "val"):
        dval = lgb.Dataset(
             X_val, label=y_val[:, i], reference=dtrain,
             weight=items["perishable"] * 0.25 + 1,
             categorical_feature=cate_vars)

        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], early_stopping_rounds=50,
            verbose_eval=100
        )

    else:

        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain], early_stopping_rounds=50, verbose_eval=100
        )

    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))

    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

    if param_1 == "val":
        val_pred.append(bst.predict(
            X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))

del X_train, y_train
##########################################################################
# Validate
# Need to use expm1 when y is log1p

if param_1 == "val":

    logger.info('validate accuracy ...')

    valid = pd.DataFrame(
            np.expm1(y_val), index=df_2017.index,
            columns=pd.date_range("2017-07-26", periods=16)
        ).stack().to_frame("unit_sales")
    valid = valid.reset_index()

    pred = pd.DataFrame(
            np.expm1(np.array(val_pred).transpose()), index=df_2017.index,
            columns=pd.date_range("2017-07-26", periods=16)
        ).stack().to_frame("pred_sales")
    pred = pred.reset_index()

    test_e = pd.merge(valid, pred, on=['item_nbr', 'store_nbr', 'level_2'])
    test_e["date"] = test_e.level_2

    del valid, pred
    del X_val, y_val
    del bst
    del X_train_allF, X_val_allF
    del df_2017

    test_e.to_pickle('../data/V034.p')

    # Check memory usage of test_e
    print(test_e.memory_usage(index=True))
    new_mem_test=test_e.memory_usage(index=True).sum()
    print("test dataset uses ",new_mem_test/ 1024**2," MB after changes")

    gc.collect()
    
    items = items.reset_index()
    test = pd.merge(test_e, items, on='item_nbr',how='inner')[['unit_sales', 'pred_sales', 'date', 'perishable']]
    del test_e, items

    # Check memory usage of test
    print(test.memory_usage(index=True))
    new_mem_test=test.memory_usage(index=True).sum()
    print("test dataset uses ",new_mem_test/ 1024**2," MB after changes")

    gc.collect()

    eval_test(test)

##########################################################################
# Submit
else:

    logger.info('Making submission...')

    y_test = np.array(test_pred).transpose()
    df_preds = pd.DataFrame(
        y_test, index=df_2017.index,
        columns=pd.date_range("2017-08-16", periods=16)
    ).stack().to_frame("unit_sales")
    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

    submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
    submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
    submission.to_csv('../submit/T034_tmp.csv', float_format='%.4f', index=None)

    # PZ, Check overral result
    print("SUM =",  submission.unit_sales.sum())
    print("MEAN =",  submission.unit_sales.mean())

    ##########################################################################
    df_prev = submission

    df_sub = pd.read_csv('../input/sub_zero_30d.csv')

    t_new = pd.merge(df_prev, df_sub, on=['id'], how='left')
    t_new['unit_sales'] = t_new.unit_sales_y.combine_first(t_new.unit_sales_x)

    submission = t_new[['id', 'unit_sales']]
    del t_new

    print("Merged  SUM =",  submission.unit_sales.sum())
    print("Merged  MEAN =",  submission.unit_sales.mean())

    submission.to_csv('../submit/T034_singleDowPromo.csv.gz',
                      float_format='%.4f', index=None, compression='gzip')
