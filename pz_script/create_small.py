import pandas as pd
import numpy as np
from datetime import timedelta
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

logger = getLogger(__name__)

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

logger.info('start')

#Load Train Data
dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32'}
#train = pd.read_csv('../input/train.csv', usecols=[1,2,3,4,5], dtype=dtypes, parse_dates=['date'] )

train = pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date'])

logger.info('load train data successful')
#train_small = train.loc[(train.item_nbr==2116416 | train.item_nbr==2113914 ),]

train_small = train.loc[(train.store_nbr==1 ),]


#Load Test Data

test = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date'])
logger.info('load test data')


# Fixed test small data on T300. It become much slower
# test_small = test.loc[(train.store_nbr==1),]

test_small = test.loc[(test.store_nbr==1),]


#Write File
test_small.to_csv('../input/test_1s.csv', index=False)
train_small.to_csv('../input/train_1s.csv', index=False)

logger.info('end')
