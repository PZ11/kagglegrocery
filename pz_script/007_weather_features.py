
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

dtype_weather={"TEMP":np.float32,
               "VISIB":np.float32,
               "PRCP": np.float32
}
    
weather = pd.read_csv('../input/Weather_20180107.csv',dtype=dtype_weather,parse_dates=["YEARMODA"],)
weather["date"] = pd.to_datetime(weather['YEARMODA'],format='%Y%m%d').dt.date
weather['ID'] = 1

t2014 = date(2014, 8, 6)
t2015 = date(2015, 8, 5)
t2016 = date(2016, 8, 3)
t2017 = date(2017, 5, 31)
train_week_2017 = 9

logger.info('Load data successful')

###############################################################################
# Functions


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range
              (dt - timedelta(days=minus), periods=periods, freq=freq)]



def prepare_dataset(t2017):
    X = pd.DataFrame({
         "ID": weather_temp.ID,
        "date": (t2017),
    })

    for i in range(16):
        for j in range(7):
            X["TEMP_{}_d{}".format(i,j)] = get_timespan(weather_temp, t2017, i+j-3, 1).values.ravel()


    for i in range(16):
        for j in range(7):
            X["VISIB_{}_d{}".format(i,j)] = get_timespan(weather_visib, t2017, i+j-3, 1).values.ravel()

            
    for i in range(16):
        for j in range(7):
            X["PRCP_{}_d{}".format(i,j)] = get_timespan(weather_prcp, t2017, i+j-3, 1).values.ravel()

    return X

###############################################################################

weather_temp = weather[['ID','date','TEMP']].set_index(
    ['ID','date'])[["TEMP"]].unstack(
        level=-1).fillna(0)
weather_temp.columns = weather_temp.columns.get_level_values(1)
weather_temp.reset_index(inplace = True)

weather_visib = weather[['ID','date','VISIB']].set_index(
    ['ID','date'])[["VISIB"]].unstack(
        level=-1).fillna(0)
weather_visib.columns = weather_visib.columns.get_level_values(1)
weather_visib.reset_index(inplace = True)


weather_prcp = weather[['ID','date','PRCP']].set_index(
    ['ID','date'])[["PRCP"]].unstack(
        level=-1).fillna(0)
weather_prcp.columns = weather_prcp.columns.get_level_values(1)
weather_prcp.reset_index(inplace = True)


weather_temp[pd.datetime(2017, 6, 28)] = 0
weather_visib[pd.datetime(2017, 6, 28)] = 0
weather_prcp[pd.datetime(2017, 6, 28)] = 0


weather_temp[pd.datetime(2016, 7, 15)] = 0
weather_visib[pd.datetime(2016, 7 , 15)] = 0
weather_prcp[pd.datetime(2016, 7, 15)] = 0

weather_temp[pd.datetime(2016, 7, 16)] = 0
weather_visib[pd.datetime(2016, 7 , 16)] = 0
weather_prcp[pd.datetime(2016, 7, 16)] = 0

weather_temp[pd.datetime(2016, 7, 17)] = 0
weather_visib[pd.datetime(2016, 7 , 17)] = 0
weather_prcp[pd.datetime(2016, 7, 17)] = 0

weather_temp[pd.datetime(2016, 7, 18)] = 0
weather_visib[pd.datetime(2016, 7 , 18)] = 0
weather_prcp[pd.datetime(2016, 7, 18)] = 0
##########################################################################

logger.info('Preparing traing dataset...')

X_l = []

# Add train data on Aug 2014 and Aug 2015


logger.info('Preparing 2015 training dataset...')
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp = prepare_dataset(
        t2015 + delta
    )
    X_l.append(X_tmp)
 
logger.info('Preparing 2017 training dataset...')
for i in range(train_week_2017):
    delta = timedelta(days=7 * i)
    X_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)


logger.info('Preparing 2016 training dataset...')
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp = prepare_dataset(
        t2016 + delta
    )
    X_l.append(X_tmp)

X_train = pd.concat(X_l, axis=0)

del X_l

delta = timedelta(0)

X_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16))
    
##########################################################################
# output   

X_train.to_pickle('../data/weather_train.p')
X_val.to_pickle('../data/weather_val.p')
X_test.to_pickle('../data/weather_test.p')
 
