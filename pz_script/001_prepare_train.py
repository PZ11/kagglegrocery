
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

from load_data import load_input_csv_data, add_missing_days

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

train_out, test_out = load_input_csv_data(param_1)

test_out = test_out.reset_index()

####################################################################################
###############################
# Find store-items that is always on promo at Wed 

train_2017 = train_out.loc[train_out['date'] > date(2017,1,1),]

train_2017_wed = train_2017.loc[(train_2017['date'].dt.dayofweek == 2), ]
train_2017_wed.onpromotion =train_2017_wed.onpromotion.astype(int)

train_2017_wed_promo = train_2017_wed[["store_nbr", "item_nbr", "date", "onpromotion"]]\
                                      .groupby(['item_nbr','store_nbr'])\
                                      .agg({'onpromotion': 'sum', 'date':'count'})\
                                      .reset_index()

train_2017_wed_promo_s_i = train_2017_wed_promo.loc[
    train_2017_wed_promo['onpromotion'] == train_2017_wed_promo['date'],]

###############################
# 
test_wed = test_out.loc[(test_out['date'].dt.dayofweek == 2), ]
test_wed = pd.merge(test_wed, train_2017_wed_promo_s_i[['store_nbr', 'item_nbr']],
                          on = ('store_nbr', 'item_nbr'), how = 'inner')


test_wed['onpromotion'] = False
test_wed = test_wed.rename(columns={'onpromotion': 'new_onpromotion'})
del test_wed['id']
## !!!! Stupid way to set onpromotion to False 
test_out_new = pd.merge(test_out, test_wed, on=('store_nbr', 'item_nbr', 'date'), how='left')
test_out_new['onpromotion'] = test_out_new.new_onpromotion.combine_first(test_out_new.onpromotion)

del test_out_new['new_onpromotion']

test_out_new= test_out_new.set_index(
        ['store_nbr', 'item_nbr', 'date'])

print( test_out.groupby('onpromotion').size()) 
print( test_out_new.groupby('onpromotion').size()) 

print( test_out.shape) 
print( test_out_new.shape) 


###############################
 
## It is very slow to use index.isin
## train_out[train_out.index.isin(train_2017_wed_a.index)]['onpromotion'] = False

# 
train_wed = train_out.loc[(train_out['date'].dt.dayofweek == 2), ]
train_wed = pd.merge(train_wed, train_2017_wed_promo_s_i[['store_nbr', 'item_nbr']],
                          on = ('store_nbr', 'item_nbr'), how = 'inner')


train_wed['onpromotion'] = False
train_wed = train_wed.rename(columns={'onpromotion': 'new_onpromotion'})
del train_wed['unit_sales']

## !!!! Stupid way to set onpromotion to False 
train_out_new = pd.merge(train_out, train_wed, on=('store_nbr', 'item_nbr', 'date'), how='left')
train_out_new['onpromotion'] = train_out_new.new_onpromotion.combine_first(train_out_new.onpromotion)

del train_out_new['new_onpromotion']


print( train_out.groupby('onpromotion').size()) 
print( train_out_new.groupby('onpromotion').size()) 

print( train_out.shape) 
print( train_out_new.shape) 




####################################################################################
if param_1 == "1s":
    train_out_new.to_pickle('../data/train_1s.p')
    test_out_new.to_pickle('../data/test_1s.p')
    
else:

    train_out_new.to_pickle('../data/train.p')
    test_out_new.to_pickle('../data/test.p')
    
