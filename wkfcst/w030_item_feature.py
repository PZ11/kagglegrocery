
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

from load_data import load_input_data, add_missing_days, add_missing_days_nopromo

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

df_train, df_test = load_input_data(param_1)

items = pd.read_csv("../input/items.csv",).set_index("item_nbr")

calendar = pd.read_csv("../input/calendar.csv",)
calendar['date'] = pd.to_datetime(calendar['salesdate'])

logger.info('Load data successful')

###############################################################################
# Functions


def get_timespan(df, dt, minus, periods):
    wk_columns = [i for i in range(dt - minus, dt - minus + periods)]
    return df[wk_columns]


def prepare_dataset(wknum, is_train=True):
    X = pd.DataFrame({
        "item_nbr": df_wk.item_nbr,
        "wknum": (wknum), 
        "item_mean_01":    get_timespan(df_wk, wknum,  1,  1).values.ravel(),
        "item_mean_02": get_timespan(df_wk, wknum,  2,  2).mean(axis=1).values,
        "item_mean_03": get_timespan(df_wk, wknum,  3,  3).mean(axis=1).values,
        "item_mean_04": get_timespan(df_wk, wknum,  4,  4).mean(axis=1).values,
        "item_mean_05": get_timespan(df_wk, wknum,  5,  5).mean(axis=1).values,
        "item_mean_06": get_timespan(df_wk, wknum,  6,  6).mean(axis=1).values,
        "item_mean_08": get_timespan(df_wk, wknum,  8,  8).mean(axis=1).values,
        "item_mean_10": get_timespan(df_wk, wknum, 10, 10).mean(axis=1).values,
        "item_mean_12": get_timespan(df_wk, wknum, 12, 12).mean(axis=1).values,
        "item_mean_16": get_timespan(df_wk, wknum, 16, 16).mean(axis=1).values,
        "item_mean_20": get_timespan(df_wk, wknum, 20, 20).mean(axis=1).values,
        
        "item_mean_p2_01": get_timespan(df_wk, wknum,  2,  1).mean(axis=1).values,
        "item_mean_p2_02": get_timespan(df_wk, wknum,  4,  2).mean(axis=1).values,
        "item_mean_p2_03": get_timespan(df_wk, wknum,  6,  3).mean(axis=1).values,
        "item_mean_p2_05": get_timespan(df_wk, wknum, 10,  5).mean(axis=1).values,
        "item_mean_p2_10": get_timespan(df_wk, wknum, 20, 10).mean(axis=1).values,      
    })

        
    if is_train:
        y_columns = [i for i in range(wknum, wknum + 2)]
        y = df_wk[y_columns].values
        return X, y
    return X


def calc_ratio(df):

    df['item_ratio_ty_p_1'] = df['item_mean_p2_01'] / df['item_mean_01']
    df['item_ratio_ty_p_2'] = df['item_mean_p2_02'] / df['item_mean_02']
    df['item_ratio_ty_p_3'] = df['item_mean_p2_03'] / df['item_mean_03']
    df['item_ratio_ty_p_5'] = df['item_mean_p2_05'] / df['item_mean_05']
    df['item_ratio_ty_p_10']= df['item_mean_p2_10'] / df['item_mean_10']
    
    del df['item_mean_p2_10'], df['item_mean_p2_05'], 
    del df['item_mean_p2_01'], df['item_mean_p2_02'], df['item_mean_p2_03']

    return df

###############################################################################

df_2017 = df_train.loc[df_train.date >= pd.datetime(2013, 5, 1)]
del df_train


# Aggregate to item level
df_train_item = df_2017[['item_nbr','date', 'store_nbr', 'unit_sales']].groupby(['item_nbr','date'])\
    .agg({'unit_sales': 'sum', 'store_nbr':'count'}).reset_index()

df_train_item["item_avg_sales"] = df_train_item["unit_sales"] / df_train_item["store_nbr"]

# Join with calendar table, aggregate to weekly
df_train_wk = pd.merge(df_train_item, calendar[['yearplusweekno', 'date']], on = 'date')
df_train_wk_sale = df_train_wk.groupby(["item_nbr", "yearplusweekno"])\
    ['unit_sales'].sum().to_frame('sales').reset_index()



df_wk_stack = df_train_wk_sale.set_index(
    ["item_nbr", "yearplusweekno"])[["sales"]].unstack(
        level=-1).fillna(0)
df_wk_stack.columns = df_wk_stack.columns.get_level_values(1)

df_wk = pd.DataFrame(df_wk_stack.copy())
df_wk.reset_index(inplace = True)

df_wk.info()
del df_train_wk_sale, df_wk_stack
gc.collect()



##########################################################################
##########################################################################
logger.info('Preparing traing dataset...')

t2014 = 201431
t2015 = 201531
t2016 = 201631
t2017 = 201722
train_week_2017 = 9

X_l, y_l = [], []

# Add train data on Aug 2014 and Aug 2015
logger.info('Preparing 2014 training dataset...')
for i in range(4):
    X_tmp, y_tmp = prepare_dataset(t2014 + i)

    X_l.append(X_tmp)
    y_l.append(y_tmp)

logger.info('Preparing 2015 training dataset...')
for i in range(4):
    X_tmp, y_tmp = prepare_dataset(t2015 + i)

    X_l.append(X_tmp)
    y_l.append(y_tmp)


logger.info('Preparing 2016 training dataset...')
for i in range(4):
    X_tmp, y_tmp = prepare_dataset(t2016 + i )

    X_l.append(X_tmp)
    y_l.append(y_tmp)


# Always load 9 weeks of data. if val, 2 weeks will be removed in 100_model. 
logger.info('Preparing 2017 training dataset...')
for i in range(train_week_2017):
    X_tmp, y_tmp = prepare_dataset( t2017 + i )

    X_l.append(X_tmp)
    y_l.append(y_tmp)

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l


X_val, y_val = prepare_dataset(201730)
X_test = prepare_dataset(201733, is_train=False)

X_train = calc_ratio(X_train).fillna(0)
X_val = calc_ratio(X_val).fillna(0)
X_test = calc_ratio(X_test).fillna(0)

print("total Train set: ", X_train.shape)
print("total val set: ", X_val.shape)


##########################################################################
logger.info('Save Item Features ...')

y_columns = ["fcst_wk" + str(i) for i in range(1, 3)]



df_y_train = pd.DataFrame(data = y_train, columns = y_columns)
X_train.reset_index(inplace = True)
X_train.reindex(index = df_y_train.index)
#train_out = pd.concat([X_train, df_y_train], axis = 1) 
train_out = X_train

df_y_val = pd.DataFrame(data = y_val, columns = y_columns)
X_val.reset_index(inplace = True)
X_val.reindex(index = df_y_val.index)
#val_out = pd.concat([X_val, df_y_val], axis = 1)
val_out = X_val



##########################################################################
# output


if param_1 == "1s":
    train_out.to_pickle('../wkdata/wk_item_train_1s.p')
    val_out.to_pickle('../wkdata/wk_item_val_1s.p')
    X_test.to_pickle('../wkdata/wk_item_test_1s.p')
    
else:
    train_out.to_pickle('../wkdata/wk_item_train.p')
    val_out.to_pickle('../wkdata/wk_item_val.p')
    X_test.to_pickle('../wkdata/wk_item_test.p')




