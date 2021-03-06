import pandas as pd
import numpy as np
from datetime import timedelta
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import sklearn.metrics as skl_metrics
import math

logger = getLogger(__name__)

pd.options.mode.chained_assignment = None  # default='warn'
DIR = 'result_tmp/'

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

def NWRMSLE(y, pred, weights=None):
    err2 = skl_metrics.mean_squared_log_error(y, pred, sample_weight=weights)
    return math.sqrt(err2)

# It will produce same result as previous one. 
def NWRMSLE_A(y, pred, weights):
    y = np.array(y)
    pred = np.array(pred)
    weights = np.array(weights)
    weighted_errors = np.dot(np.square(np.log1p(pred) - np.log1p(y)), np.transpose(weights))
    weights_sum = np.sum(weights)
    return math.sqrt(weighted_errors/weights_sum)




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

    
    test_e['error'] =  abs(test_e.pred_sales - test_e.unit_sales)
    print("Bias =",  (test_e.pred_sales.sum() - test_e.unit_sales.sum()) /  test_e.unit_sales.sum())
    print("WMAPE =",  abs(test_e.error.sum() - test_e.unit_sales.sum()) /  test_e.unit_sales.sum())


#--------------------------------------------------------------------------------------------------

logger.info('start')

dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32'}
train_all = pd.read_csv('../input/train.csv', usecols=[1,2,3,4,5], dtype=dtypes, parse_dates=['date'],
                    skiprows=range(1, 86672217) #Skip dates before 2016-08-01
                    )
items = pd.read_csv('../input/items.csv'  )

# Test period 07-26 to 08-10, Total row in  train_small 35328
#train_all = pd.read_csv('../input/train_small.csv',  dtype=dtypes, parse_dates=['date'],
#                    skiprows=range(1, 26565) #Skip dates before 2016-08-01
#                    )
logger.info('load data successful')


train = train_all.loc[(train_all.date <= '2017-07-25'), ]
test = train_all.loc[(train_all.date > '2017-07-25') & (train_all.date <= '2017-08-10' ), ]


#--------------------------------------------------------------------------------------------------

train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
train['dow'] = train['date'].dt.dayofweek

# creating records for all items, in all markets on all dates
# for correct calculation of daily unit sales averages.
u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()
train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)
train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
)

logger.info('reindex train data')

del u_dates, u_stores, u_items

train.loc[:, 'unit_sales'].fillna(0, inplace=True) # fill NaNs
train.reset_index(inplace=True) # reset index and restoring unique columns  
lastdate = train.iloc[train.shape[0]-1].date

#Days of Week Means
#By tarobxl: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/42948
ma_dw = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw')
ma_dw.reset_index(inplace=True)
ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk')
ma_wk.reset_index(inplace=True)

#Moving Averages
logger.info('start calcualte moving average')
ma_is = train[['item_nbr','store_nbr','unit_sales']].groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais226')
for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')

del tmp,tmpg,train

ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)

#---------------------------------------------------------------------------------------
#Load test

logger.info('load test data')

test['dow'] = test['date'].dt.dayofweek
test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])

del ma_is, ma_wk, ma_dw


#Forecasting Test
test['pred_sales'] = test.mais 
pos_idx = test['mawk'] > 0
test_pos = test.loc[pos_idx]
test.loc[pos_idx, 'pred_sales'] = test_pos['mais'] * test_pos['madw'] / test_pos['mawk']
test.loc[:, "pred_sales"].fillna(0, inplace=True)
test['pred_sales'] = test['pred_sales'].apply(pd.np.expm1) # restoring unit values 

test.loc[:, "unit_sales"].fillna(0, inplace=True)
test.loc[(test.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
test.loc[(test.pred_sales<0),'pred_sales'] = 0 # eliminate negatives

#50% more for promotion items
test.loc[test['onpromotion'] == True, 'unit_sales'] *= 1.25

eval_test(test)

logger.info('end')
