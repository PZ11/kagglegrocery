
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding
more average features and weekly average features on it.

Run 1 Store validation  : .py 1s 045
Run 1 Store submission  : .py 1ss 045
Run all Store validation: .py val 045
Run all Store submission: .py a 045


"""
from datetime import date

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys

import gc

#from nwrmsle_eval import eval_test

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

print("input parameter = ", param_1)
print("Test/val number = ", param_2)

exp_filename = '../wkdata/V' + param_2 + '.p'
logger.info(exp_filename)

# Comment out Val Step, no time to validate 
if ((param_1 == "1ss") or (param_1 == "1s")):
    train_out = pd.read_pickle('../wkdata/wk_storeitem_train_1s.p')
    #val_out = pd.read_pickle('../wkdata/wk_storeitem_val_1s.p')
    X_test_out = pd.read_pickle('../wkdata/wk_storeitem_test_1s.p')

    item_train_out = pd.read_pickle('../wkdata/wk_item_train_1s.p')
    #item_val_out = pd.read_pickle('../wkdata/wk_item_val_1s.p')
    item_X_test_out = pd.read_pickle('../wkdata/wk_item_test_1s.p')

    df_test = pd.read_csv(
        "../input/test_1s.csv", usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]  # , date_parser=parser
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )

else:
    train_out = pd.read_pickle('../wkdata/wk_storeitem_train.p')
    val_out = pd.read_pickle('../wkdata/wk_storeitem_val.p')
    X_test_out = pd.read_pickle('../wkdata/wk_storeitem_test.p')

    item_train_out = pd.read_pickle('../wkdata/wk_item_train.p')
    item_val_out = pd.read_pickle('../wkdata/wk_item_val.p')
    item_X_test_out = pd.read_pickle('../wkdata/wk_item_test.p')
    
    df_test = pd.read_csv(
        "../input/test.csv", usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]  # , date_parser=parser
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )


# On validation step, need remove last 2 weeks in the train data
# if ((param_1 == "val") or (param_1 == "1s")):
#    train_out = train_out.loc[train_out["wknum"] < 201730, ]

items = pd.read_csv("../input/items.csv",)

#items_val = pd.read_csv("../input/items.csv",).set_index("item_nbr")
#items_val = items_val.reindex(val_out['item_nbr'])

logger.info('Load data successful')


###############################################################################
# Delete index columns before merge
del train_out["index"]


########################################
# Merge item features
del item_train_out["index"]


train_out = pd.merge(train_out, item_train_out, how='inner', on=['item_nbr', 'wknum'])
#val_out = pd.merge(val_out, item_val_out, how='inner', on=['item_nbr', 'wknum'])
X_test_out = pd.merge(X_test_out, item_X_test_out, how='inner', on=['item_nbr', 'wknum'])

del item_train_out, item_X_test_out
gc.collect()



###############################################################################
logger.info('Preparing traing dataset...')

all_columns = train_out.columns.tolist()

y_columns = ['fcst_wk'+str(i) for i in range(1, 3)]
x_columns = [item for item in all_columns if item not in y_columns]

features_all = x_columns
features_all.remove("wknum") 
features_all.remove("item_nbr") 
features_all.remove("store_nbr") 

X_train_out = train_out[x_columns]
#X_val_out = val_out[x_columns]

y_train = train_out[y_columns].values
#y_val = val_out[y_columns].values

X_train_allF = X_train_out[features_all]
#X_val_allF = X_val_out[features_all]
X_test_allF = X_test_out[features_all]

items = items.set_index("item_nbr")
items = items.reindex(train_out.item_nbr)

del train_out
del X_train_out
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
#val_pred = []
test_pred = []
cate_vars = []


train_week_2017 = 9

  
features_all = X_train_allF.columns.tolist()

for i in range(2):
    print("=" * 70)
    logger.info("Step %d" % (i+1))
    print("=" * 70)
    features_t = features_all.copy()


    X_train = X_train_allF[features_t]
    #X_val = X_val_allF[features_t]
    X_test = X_test_allF[features_t]


    
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]]) * 0.25 + 1
    )
    '''
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
    '''
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


del X_train, y_train
del dtrain
gc.collect()

##########################################################################
# Export data, no time for validate 
# Need to use expm1 when y is log1p

# Export Files
logger.info('Making submission...')

y_columns = ["fcst_wk" + str(i) for i in range(1, 3)]

X_test_out.reset_index(inplace = True)
del X_test_out["wknum"]
X_test_out = X_test_out.set_index(["store_nbr", "item_nbr"])

y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    np.expm1(np.array(y_test)), index=X_test_out.index,
    columns=y_columns
).stack().to_frame("unit_sales")

df_preds.to_pickle(exp_filename)
