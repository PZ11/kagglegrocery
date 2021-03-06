import pandas as pd
import numpy as np
from datetime import timedelta
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import sklearn.metrics as skl_metrics
import xgboost as xgb
from sklearn.cross_validation import train_test_split

import math

logger = getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'

DIR = './result_tmp/'

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

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def NWRMSLE(y, pred, weights=None):
    err2 = skl_metrics.mean_squared_log_error(y, pred, sample_weight=weights)
    return math.sqrt(err2)

def NWRMSLE_A(y, pred, weights):
    y = np.array(y)
    pred = np.array(pred)
    weights = np.array(weights)
    weighted_errors = np.dot(np.square(np.log1p(pred) - np.log1p(y)), np.transpose(weights))
    weights_sum = np.sum(weights)
    return math.sqrt(weighted_errors/weights_sum)

def NWRMSLE_lgb(pred, dtrain):
    y = list(dtrain.get_label())
    score = NWRMSLE(y, pred)
    return 'NWRMSLE', score, False

def eval_test(test_e):

    test_e['weights'] = 1
    test_e.loc[(test_e.perishable == 1), ('weights')] = 1.25

    result = NWRMSLE(test_e.unit_sales.astype(np.float64),test_e.pred_sales.astype(np.float64), test_e.weights)

    print("Eval All, Number of rows in test is", test_e.shape[0])
    print("Eval all, Forecast Period From:", min(test_e.date)," To: ", max(test_e.date))

    #### check result on first 6 days.
    test_p1 = test_e.loc[(test_e.date < '2017-08-01'), ]
    result_p1 = NWRMSLE_A(test_p1.unit_sales.astype(np.float32),test_p1.pred_sales.astype(np.float32), test_p1.weights)

    print("Eval P1, Number of rows in test is", test_p1.shape[0])
    print("Eval P1, Forecast Period From:", min(test_p1.date)," To: ", max(test_p1.date))

    #### check result on last 10 days.
    test_p2 = test_e.loc[(test_e.date >= '2017-08-01'), ]
    result_p2 = NWRMSLE_A(test_p2.unit_sales.astype(np.float32),test_p2.pred_sales.astype(np.float32), test_p2.weights)

    print("Eval P2, Number of rows in test is", test_p2.shape[0])
    print("Eval P2, Forecast Period From:", min(test_p2.date)," To: ", max(test_p2.date))

    print("Eval All Weighted NWRMSLE = ",result)
    print("Eval P1  Weighted NWRMSLE = ",result_p1)
    print("Eval P2  Weighted NWRMSLE = ",result_p2)

#--------------------------------------------------------------------------------------------------

logger.info('start')

items = pd.read_csv('../input/items.csv'  )

dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32'}
train_all = pd.read_csv('../input/train.csv', usecols=[1,2,3,4,5], dtype=dtypes, parse_dates=['date'] )
#train_all = pd.read_csv('../input/train_small.csv', usecols=[1,2,3,4,5], dtype=dtypes, parse_dates=['date'] )

df_train = train_all.loc[((train_all.date >= '2016-06-01') & (train_all.date <= '2016-08-31' ))
                         |((train_all.date >= '2017-06-01') & (train_all.date <= '2017-08-31' )) , ]
del train_all

logger.info('load data successful')

#train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion

df_train.loc[(df_train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
df_train.loc[:, 'unit_sales'].fillna(0, inplace=True) # fill NaNs

df_train['DOW'] = df_train['date'].dt.dayofweek
df_train['WOY'] = df_train['date'].dt.weekofyear
df_train['Year'] = df_train['date'].dt.year
df_train['Month'] = df_train['date'].dt.month
df_train['Day'] = df_train['date'].dt.day

print('training data prepared')

#-------------------------------------------------------------------------------------------------

train_m06 = df_train.loc[(df_train.Month == 6),]
    
ma_m06 = train_m06[['item_nbr','store_nbr','Year','unit_sales']].groupby(['item_nbr','store_nbr','Year'])\
['unit_sales'].mean().to_frame('avg_m06')
ma_m06.reset_index(inplace=True)

df_train = pd.merge(df_train, ma_m06, how='left', on=['item_nbr','store_nbr','Year'])

#features = (['DOW', 'WOY', 'Month', 'Day', 'avg_m06'])
#print(features)

#-------------------------------------------------------------------------------------------------

train_m07 = df_train.loc[(df_train.Month == 7),]
    
ma_m07 = train_m07[['item_nbr','store_nbr','Year','unit_sales']].groupby(['item_nbr','store_nbr','Year'])\
['unit_sales'].mean().to_frame('avg_m07')
ma_m07.reset_index(inplace=True)
df_train = pd.merge(df_train, ma_m07, how='left', on=['item_nbr','store_nbr','Year'])

features = (['DOW', 'WOY', 'Month', 'Day', 'avg_m06', 'avg_m07'])

#--------------------------------------------------------------------------------------------------
train = df_train.loc[(df_train.date >= '2016-07-01') & (df_train.date <= '2016-08-31' ), ]
test = df_train.loc[(df_train.date > '2017-07-25') & (df_train.date <= '2017-08-10' ), ]


print('training data processed')

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

print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.5, random_state=10)
y_train = np.log1p(X_train.unit_sales)
y_valid = np.log1p(X_valid.unit_sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model_xgb = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=20, verbose_eval=True)


create_feature_map(features)
importance = model_xgb.get_fscore(fmap='xgb.fmap')
print(importance)


#-------------------------------------------------------------------------------------
#Load test
#test = valid
test['pred_sales'] = np.exp(model_xgb.predict(xgb.DMatrix(test[features])))


#---------------------- test_e to evaluate the result --------------------------------
#weights = np.ones(test.shape[0])
test_e = pd.merge(test, items, on='item_nbr',how='inner')
eval_test(test_e)

test_e['error'] =  abs(test_e.pred_sales - test_e.unit_sales)
print("Bias =",  (test_e.pred_sales.sum() - test_e.unit_sales.sum()) /  test_e.unit_sales.sum())
print("WMAPE =",  abs(test_e.error.sum() - test_e.unit_sales.sum()) /  test_e.unit_sales.sum())

#-------------------------------------------------------------------------------------
logger.info('end')
