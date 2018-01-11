import pandas as pd
import numpy as np
from datetime import date, timedelta

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

##########################################################################

def load_input_data(param_1):

        if param_1 == "1s":
            df_train = pd.read_pickle('../data/train_1s.p')
            df_test= pd.read_pickle('../data/test_1s.p')

        else:

            df_train = pd.read_pickle('../data/train.p')
            df_test = pd.read_pickle('../data/test.p')
            
        return df_train, df_test


def add_missing_days(df_2017, promo_2017,param_1):
    
    df_2017[pd.datetime(2016, 12, 25)] = 0
    df_2017[pd.datetime(2015, 12, 25)] = 0
    df_2017[pd.datetime(2014, 12, 25)] = 0
    df_2017[pd.datetime(2013, 12, 25)] = 0
    promo_2017[pd.datetime(2017, 9, 1)] = 0
    promo_2017[pd.datetime(2017, 9, 2)] = 0
    promo_2017[pd.datetime(2017, 9, 3)] = 0

    if param_1 == "1s":
        df_2017[pd.datetime(2017, 1, 1)] = 0
        df_2017[pd.datetime(2016, 1, 1)] = 0
        df_2017[pd.datetime(2015, 1, 1)] = 0    
        df_2017[pd.datetime(2015, 7, 7)] = 0
        df_2017[pd.datetime(2014, 1, 1)] = 0
        df_2017[pd.datetime(2013, 1, 1)] = 0
        promo_2017[pd.datetime(2015, 7, 7)] = 0


    return df_2017, promo_2017

def add_missing_days_nopromo(df_2017, param_1):
    
    df_2017[pd.datetime(2016, 12, 25)] = 0
    df_2017[pd.datetime(2015, 12, 25)] = 0
    df_2017[pd.datetime(2014, 12, 25)] = 0
    df_2017[pd.datetime(2013, 12, 25)] = 0
 
    if param_1 == "1s":
        df_2017[pd.datetime(2017, 1, 1)] = 0
        df_2017[pd.datetime(2016, 1, 1)] = 0
        df_2017[pd.datetime(2015, 1, 1)] = 0    
        df_2017[pd.datetime(2015, 7, 7)] = 0
        df_2017[pd.datetime(2014, 1, 1)] = 0
        df_2017[pd.datetime(2013, 1, 1)] = 0
    return df_2017



def load_input_csv_data(param_1):

    if param_1 == "1s":
        TRAIN_DATA = '../input/train_1s.csv'
        TEST_DATA = '../input/test_1s.csv'
    else: 
        TRAIN_DATA = '../input/train.csv'
        TEST_DATA = '../input/test.csv'
        
    dtype_dict={"id":np.uint32,
                "store_nbr":np.uint8,
                "item_nbr":np.uint32,
                "unit_sales":np.float32,
                "onpromotion": bool
               }


    df_train = pd.read_csv( TRAIN_DATA, usecols=[1, 2, 3, 4, 5],
        dtype=dtype_dict,
        converters={'unit_sales': lambda u: np.log1p(
            float(u)) if float(u) > 0 else 0},
        parse_dates=["date"],
    )
    
    df_train['unit_sales'] = df_train['unit_sales'].astype(np.float32)

    df_test = pd.read_csv(TEST_DATA,
        usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]  # , date_parser=parser
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )

    return df_train, df_test

