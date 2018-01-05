
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding
more average features and weekly average features on it.

Run 1 Store validation  : .py 1s 045
Run 1 Store submission  : .py 1ss 045
Run all Store validation: .py val 045
Run all Store submission: .py a 045


"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import math
import gc
import sklearn.metrics as skl_metrics
from sklearn.metrics import mean_squared_error

from nwrmsle_eval import eval_test

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
# 1s : validate one store data
# 1ss: submit one store
# val: validate all data
# No parameter: submit all data 

if len(sys.argv) == 1:
    param_1 = "Full Run"
    param_2 = "999"
elif len(sys.argv) == 2:
    param_1 = sys.argv[1]
    param_2 = 'tmp'
else:
    param_1 = sys.argv[1]
    param_2 = sys.argv[2]

logger.info("input parameter = ", param_1)
logger.info("Test/val number = ", param_2)
submit_filename = '../submit/T' + param_2 + '.csv.gz'
val_filename = '../data/V' + param_2 + '.p'

logger.info(submit_filename)
logger.info(val_filename)

if  ((param_1 == "1ss") or (param_1 == "1s")):
    train_out = pd.read_pickle('../data/storeitem_train_1s.p')
    val_out = pd.read_pickle('../data/storeitem_val_1s.p')
    X_test_out = pd.read_pickle('../data/storeitem_test_1s.p')

    item_train_out = pd.read_pickle('../data/item_train_1s.p')
    item_val_out = pd.read_pickle('../data/item_val_1s.p')
    item_X_test_out = pd.read_pickle('../data/item_test_1s.p')

    store_train_out = pd.read_pickle('../data/store_train_1s.p')
    store_val_out = pd.read_pickle('../data/store_val_1s.p')
    store_X_test_out = pd.read_pickle('../data/store_test_1s.p')

    s_f_train_out = pd.read_pickle('../data/storefamily_train_1s.p')
    s_f_val_out = pd.read_pickle('../data/storefamily_val_1s.p')
    s_f_X_test_out = pd.read_pickle('../data/storefamily_test_1s.p')
    
    df_test = pd.read_csv(
        "../input/test_1s.csv", usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]  # , date_parser=parser
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )
    
else:
    train_out = pd.read_pickle('../data/storeitem_train.p')
    val_out = pd.read_pickle('../data/storeitem_val.p')
    X_test_out = pd.read_pickle('../data/storeitem_test.p')

    item_train_out = pd.read_pickle('../data/item_train.p')
    item_val_out = pd.read_pickle('../data/item_val.p')
    item_X_test_out = pd.read_pickle('../data/item_test.p')

    store_train_out = pd.read_pickle('../data/store_train.p')
    store_val_out = pd.read_pickle('../data/store_val.p')
    store_X_test_out = pd.read_pickle('../data/store_test.p')

    s_f_train_out = pd.read_pickle('../data/storefamily_train.p')
    s_f_val_out = pd.read_pickle('../data/storefamily_val.p')
    s_f_X_test_out = pd.read_pickle('../data/storefamily_test.p')

    df_test = pd.read_csv(
        "../input/test.csv", usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]  # , date_parser=parser
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )


# On validation step, need remove last 2 weeks in the train data
# train_out["date"] = pd.to_datetime(train_out["date"])
#if ((param_1 == "val") or (param_1 == "1s")):
#    train_out = train_out.loc[train_out["date"] < '2017-07-19', ]

  
items = pd.read_csv(
    "../input/items.csv",
)

items_val = pd.read_csv(
    "../input/items.csv",
).set_index("item_nbr")
items_val = items_val.reindex(val_out['item_nbr'])

logger.info('Load data successful')

###############################################################################
# Delete index columns before merge 
del train_out["index"]
del item_train_out["index"]
del store_train_out["index"]
del s_f_train_out["index"]



# Merge item features 
train_out = pd.merge(train_out, item_train_out, how='inner', on=['item_nbr','date'])
val_out = pd.merge(val_out, item_val_out, how='inner', on=['item_nbr','date'])
X_test_out = pd.merge(X_test_out, item_X_test_out, how='inner', on=['item_nbr','date'])

del item_train_out, item_val_out, item_X_test_out
gc.collect()



# Merge store features
train_out = pd.merge(train_out, store_train_out, how='inner', on=['store_nbr','date'])
val_out = pd.merge(val_out, store_val_out, how='inner', on=['store_nbr','date'])
X_test_out = pd.merge(X_test_out, store_X_test_out, how='inner', on=['store_nbr','date'])

print(train_out.groupby(['date']).size())

del store_train_out, store_val_out, store_X_test_out
gc.collect()


###############################################################################
logger.info('Preparing traing dataset...')
    

all_columns = train_out.columns.tolist()

y_columns = ['day'+str(i) for i in range(1, 17)]
x_columns = [item for item in all_columns if item not in y_columns]

features_all = x_columns
features_all.remove("date") 
features_all.remove("item_nbr") 
features_all.remove("store_nbr") 

X_train_out = train_out[x_columns]
X_val_out = val_out[x_columns]

y_train = train_out[y_columns].values
y_val = val_out[y_columns].values

X_train_allF = X_train_out[features_all]
X_val_allF = X_val_out[features_all]
X_test_allF = X_test_out[features_all]

items = items.set_index("item_nbr")
items = items.reindex(train_out.item_nbr)

del train_out
del X_train_out, X_val_out
gc.collect()

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


train_week_2017 = 7
if param_1 != "val":
    train_week_2017 = 9
    
features_all = X_train_allF.columns.tolist()

for i in range(16):
    print("=" * 70)
    logger.info("Step %d" % (i+1))
    print("=" * 70)
    features_t = features_all.copy()

    for j in range(16):
        if j != i:
            features_t.remove('ly_1d_d{}'.format(j))
            features_t.remove('l2y_1d_d{}'.format(j))

    for j in range(7):
        if j != i%7:
            features_t.remove('dow_1_{}_mean'.format(j))
            features_t.remove('dow_4_{}_mean'.format(j))
            features_t.remove('dow_8_{}_mean'.format(j))
            features_t.remove('dow_13_{}_mean'.format(j))
            features_t.remove('dow_26_{}_mean'.format(j))
            features_t.remove('dow_52_{}_mean'.format(j))
            features_t.remove('dow_ly3w_{}_mean'.format(j))
            features_t.remove('dow_ly8w_{}_mean'.format(j))
            
            features_t.remove('item_dow_4_{}_mean'.format(j))
            features_t.remove('item_dow_13_{}_mean'.format(j))
            features_t.remove('item_dow_26_{}_mean'.format(j))
            features_t.remove('item_dow_52_{}_mean'.format(j))

            features_t.remove('store_dow_4_{}_mean'.format(j))
            features_t.remove('store_dow_13_{}_mean'.format(j))
            features_t.remove('store_dow_26_{}_mean'.format(j))
            features_t.remove('store_dow_52_{}_mean'.format(j))     

    X_train = X_train_allF[features_t]
    X_val = X_val_allF[features_t]
    X_test = X_test_allF[features_t]
	
     
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]]) * 0.25 + 1
    )

    if ((param_1 == "val") or (param_1 == "1s")):
        dval = lgb.Dataset(
             X_val, label=y_val[:, i], reference=dtrain,
             weight=items_val["perishable"] * 0.25 + 1,
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

    logger.info("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))

    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

    if ((param_1 == "val") or (param_1 == "1s")):
        val_pred.append(bst.predict(
            X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))

del X_train, y_train
del dtrain
gc.collect()

##########################################################################
# Validate
# Need to use expm1 when y is log1p

if ((param_1 == "val") or (param_1 == "1s")):

    logger.info('validate accuracy ...')

    val_out.reset_index(inplace = True)
    val_out = val_out.set_index(["store_nbr", "item_nbr", "date"])

    valid = pd.DataFrame(
            np.expm1(y_val), index=val_out.index,
            columns=pd.date_range("2017-07-26", periods=16)
        ).stack().to_frame("unit_sales")
    valid = valid.reset_index()

    pred = pd.DataFrame(
            np.expm1(np.array(val_pred).transpose()), index=val_out.index,
            columns=pd.date_range("2017-07-26", periods=16)
        ).stack().to_frame("pred_sales")
    pred = pred.reset_index()


    test_e = pd.merge(valid, pred, on=['item_nbr', 'store_nbr', 'date', 'level_3'])
    del  test_e["date"]
    test_e["date"] = test_e.level_3


    del valid, pred
    del X_val, y_val
    del bst, dval
    del X_train_allF, X_val_allF
    gc.collect()

    test_e.to_pickle(val_filename)

    # Check memory usage of test_e
    print(test_e.memory_usage(index=True))
    new_mem_test=test_e.memory_usage(index=True).sum()
    print("test dataset uses ",new_mem_test/ 1024**2," MB after changes")

    gc.collect()
    items = pd.read_csv("../input/items.csv")
    test = pd.merge(test_e, items, on='item_nbr',how='inner')[['unit_sales', 'pred_sales', 'date', 'perishable']]

    del test_e, items
    gc.collect()

    # Check memory usage of test
    print(test.memory_usage(index=True))
    new_mem_test=test.memory_usage(index=True).sum()
    print("test dataset uses ",new_mem_test/ 1024**2," MB after changes")


    eval_test(test)

##########################################################################
# Submit
else:

    logger.info('Making submission...')

    X_test_out.reset_index(inplace = True)
    del X_test_out["date"]
    X_test_out = X_test_out.set_index(["store_nbr", "item_nbr"])

    y_test = np.array(test_pred).transpose()
    df_preds = pd.DataFrame(
        y_test, index=X_test_out.index,
        columns=pd.date_range("2017-08-16", periods=16)
    ).stack().to_frame("unit_sales")

    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

    submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
    
    submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
#    submission.to_csv('../submit/T043_tmp.csv', float_format='%.4f', index=None)

    # PZ, Check overral result
    logger.info("SUM =",  submission.unit_sales.sum())
    logger.info("MEAN =",  submission.unit_sales.mean())

    ##########################################################################
    df_prev = submission

    df_sub = pd.read_csv('../input/sub_zero_30d.csv')

    t_new = pd.merge(df_prev, df_sub, on=['id'], how='left')
    t_new['unit_sales'] = t_new.unit_sales_y.combine_first(t_new.unit_sales_x)

    submission = t_new[['id', 'unit_sales']]
    del t_new

    logger.info("Merged  SUM =",  submission.unit_sales.sum())
    logger.info("Merged  MEAN =",  submission.unit_sales.mean())

    submission.to_csv(submit_filename,
                      float_format='%.4f', index=None, compression='gzip')
