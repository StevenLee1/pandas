import pandas as pd
import multiprocessing
from time import time
import datetime


identity_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/train_identity.csv" #
transaction_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/train_transaction.csv" #(590540, 394)

test_identity_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/test_identity.csv"
test_transaction_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/test_transaction.csv"

sample_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/sample_submission.csv"

files = [test_identity_file, test_transaction_file,
         identity_file, transaction_file, sample_file]


def load_data(file):
    return pd.read_csv(file)


# Loading all datasets using multiprocessing. This speads up a process a bit.
pool = multiprocessing.Pool(processes=4)
test_id, test_tr, train_id, train_tr, sub = pool.map(load_data, files)

# merge data
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
del test_id, test_tr, train_id, train_tr


startdate = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')


