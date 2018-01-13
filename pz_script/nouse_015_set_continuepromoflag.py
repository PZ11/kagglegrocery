
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

from load_data import load_input_data, add_missing_days

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
logger.info('start')


if len(sys.argv) == 1:
    param_1 = "Full Run"
else:
    param_1 = sys.argv[1]
    print("input parameter = ", param_1)

train_out, test_out = load_input_data(param_1)
test_out = test_out.reset_index()

####################################################################################
###############################
# Load store-items that has promo more than 75% of the sales week. 

reset_promo = pd.read_csv("../input/Reset_Flag_ContinuePromo.csv",)
reset_promo['new_onpromotion'] = False

###############################
## !!!! Stupid way to set onpromotion to False 

test_out_new = pd.merge(test_out, reset_promo, on=('store_nbr', 'item_nbr'), how='left')
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

## !!!! Stupid way to set onpromotion to False 
train_out_new = pd.merge(train_out, reset_promo, on=('store_nbr', 'item_nbr'), how='left')
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
    
