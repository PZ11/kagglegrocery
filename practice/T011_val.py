
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta
import pandas as pd
import numpy as np
import sklearn.metrics as skl_metrics
from sklearn.metrics import mean_squared_error
import sys
import math

#------------------------------------------------------------------------------------------#
# Functions

def NWRMSLE(y, pred, weights=None):
    err2 = skl_metrics.mean_squared_log_error(y, pred, sample_weight=weights)
    return math.sqrt(err2)

def eval_test(test_e):

    test_e['weights'] = 1
    test_e.loc[(test_e.perishable == 1), ('weights')] = 1.25

    result = NWRMSLE(test_e.unit_sales.astype(np.float64),test_e.pred_sales.astype(np.float64), test_e.weights)

    print("Eval All, Number of rows in test is", test_e.shape[0])
    print("Eval all, Forecast Period From:", min(test_e.date)," To: ", max(test_e.date))

    #### check result on first 6 days.
    test_p1 = test_e.loc[(test_e.date < '2017-08-01'), ]
    result_p1 = NWRMSLE(test_p1.unit_sales.astype(np.float32),test_p1.pred_sales.astype(np.float32), test_p1.weights)

    print("Eval P1, Number of rows in test is", test_p1.shape[0])
    print("Eval P1, Forecast Period From:", min(test_p1.date)," To: ", max(test_p1.date))

    #### check result on last 10 days.
    test_p2 = test_e.loc[(test_e.date >= '2017-08-01'), ]
    result_p2 = NWRMSLE(test_p2.unit_sales.astype(np.float32),test_p2.pred_sales.astype(np.float32), test_p2.weights)

    print("Eval P2, Number of rows in test is", test_p2.shape[0])
    print("Eval P2, Forecast Period From:", min(test_p2.date)," To: ", max(test_p2.date))

    print("Eval All Weighted NWRMSLE = ",result)
    print("Eval P1  Weighted NWRMSLE = ",result_p1)
    print("Eval P2  Weighted NWRMSLE = ",result_p2)
    
    test_e['error'] =  abs(test_e.pred_sales - test_e.unit_sales)
    print("Bias =",  (test_e.pred_sales.sum() - test_e.unit_sales.sum()) /  test_e.unit_sales.sum())
    print("WMAPE =",  abs(test_e.error.sum() - test_e.unit_sales.sum()) /  test_e.unit_sales.sum())

    print("SUM =",  test_e.unit_sales.sum())
    print("MEAN =",  test_e.unit_sales.mean())

    print( mean_squared_error(np.log1p(test_e.unit_sales),np.log1p(test_e.pred_sales)))    
    print(result)
    print(result_p1)
    print(result_p2)
    print((test_e.pred_sales.sum() - test_e.unit_sales.sum()) /  test_e.unit_sales.sum())
    print(abs(test_e.error.sum() - test_e.unit_sales.sum()) /  test_e.unit_sales.sum())
    print(test_e.pred_sales.sum())
    print(test_e.pred_sales.mean())
    
#------------------------------------------------------------------------------------------#

#col = ['order_id', 'user_id', 'product_id', 'order_number', 'order_number_rev']
val = pd.read_pickle('../practice/data/T006_lgb_val.p')
p_val = pd.read_pickle('./data/T011_p_val.p')

print("T006 pred_sales:", val['pred_sales'].sum())
print("T006 Validation mse:", mean_squared_error(
    np.log1p(val.unit_sales),np.log1p(val.pred_sales)))

val_new = pd.merge(val, p_val,  on=['item_nbr','store_nbr', 'date'], how = 'left')

val_new['pred_sales'] = val_new.pred_sales_y.combine_first(val_new.pred_sales_x)
val_new['unit_sales'] = val_new['unit_sales_x']

print("T011 pred_sales:", val_new['pred_sales'].sum())

print("T011 Validation mse:", mean_squared_error(
    np.log1p(val_new.unit_sales),np.log1p(val_new.pred_sales)))

items = pd.read_csv('../input/items.csv'  )
test = pd.merge(val_new, items, on='item_nbr',how='inner')

eval_test(test)


