# -*- coding: utf-8 -*-
#@Author: saili
#@Time: 2019/12/2 18:51
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc



def add_column(x, colvalue_list, colname):
    value = x[colname]
    index = colvalue_list.index(value)
    value_list = []
    for i in range(len(colvalue_list)):
        value_list.append(0)
    value_list[index] = 1
    for index, value in enumerate(value_list):
        col_name = colname + '_' + str(index)
        x[col_name] = value
    return x




def one_hot(data, colname):
    assert type(data) == pd.DataFrame
    columns = data.columns
    if colname not in columns:
        print "data has no column :" + colname
        return
    colvalue_list = data[colname].unique().tolist()
    # lambda 性能不好
    data.apply(lambda x: add_column(x, colvalue_list, colname), axis = 1)
    return data

def new_column_list(colname, length):
    colname_list = []
    for i in range(length):
        colname_list.append(colname + "_" + str(i))
    return colname_list


def one_hot_encoder(data, colname, drop_col = False):
    assert type(data) == pd.DataFrame
    columns = data.columns
    if colname not in columns:
        print "data has no column :" + colname
        return
    dataset_len = len(data)
    enc = OneHotEncoder()
    colvalue_list = data[colname].unique()
    print "unique value of " + colname + " is: "
    print colvalue_list
    length = len(colvalue_list)
    colvalue_list = colvalue_list.reshape(length, 1)
    enc.fit(colvalue_list)
    result = enc.transform(data[colname].values.reshape(dataset_len, 1)).toarray()
    new_columns = new_column_list(colname, length)
    transformed_df = pd.DataFrame(data= result, columns=new_columns)
    mixed_df = pd.concat([data, transformed_df], axis=1)
    if drop_col:
        mixed_df = mixed_df.drop(columns=[colname])
    return mixed_df


def standard_scale(data, colname, drop_col=False, sigma=3):
    assert type(data) == pd.DataFrame
    columns = data.columns
    if colname not in columns:
        print "data has no column :" + colname
        return
    print "colname is :" + colname
    dataset_len = len(data)
    enc = StandardScaler()
    value_array = data[colname].values.reshape(dataset_len, 1)
    enc.fit(value_array)
    mean_value = enc.mean_[0]
    print "mean_value is :" + str(mean_value)
    std_value = enc.scale_[0]
    print "std_value is :" + str(std_value)
    # 将空值转换为均值
    data.loc[data[colname].isnull(), colname] = mean_value
    # 将超过sigma准则的值视为异常值
    data.loc[abs(data[colname]-mean_value) > sigma * std_value, colname] = mean_value
    # 归一化
    scaled_data = enc.transform(data[colname].values.reshape(dataset_len, 1))
    new_column_name = [colname + "_scale"]
    transformed_df = pd.DataFrame(data=scaled_data, columns=new_column_name)
    mixed_df = pd.concat([data, transformed_df], axis=1)
    if drop_col:
        mixed_df = mixed_df.drop(columns=[colname])
    return mixed_df


def reduce_mem_usage(df):
    assert type(df) == pd.DataFrame
    print "before save memory, df info is "
    print df.info()
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    print "after save memory, df info is "
    print df.info()
    return df


def id_split(dataframe):
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]

    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]

    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]

    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]

    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[
                                                 dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1
    gc.collect()

    return dataframe




identity_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/train_identity.csv" #
transaction_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/train_transaction.csv" #(590540, 394)

test_identity_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/test_identity.csv"
test_transaction_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/test_transaction.csv"

sample_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/sample_submission.csv"

print "loading data..."
train_identity = pd.read_csv(identity_file, index_col='TransactionID')
print "successfully loaded train_identity!"
train_transaction = pd.read_csv(transaction_file, index_col="TransactionID")
print "successfully loaded train_transaction"
test_identity = pd.read_csv(test_identity_file, index_col="TransactionID")
print "successfully loaded test_identity"
test_transaction = pd.read_csv(test_transaction_file, index_col='TransactionID')
print "successfully loaded test_transaction"

train_identity = id_split(train_identity)
test_identity = id_split(test_identity)

print "merging data"
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
print "data was successfully merged"

del train_identity, train_transaction, test_identity, test_transaction

print "Train dataset has %s rows and %s columns" % (train.shape[0], train.shape[1])
print "Test dataset has %s rows and %s columns" % (test.shape[0], test.shape[1])

gc.collect()

useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3',
                   'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',
                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5',
                   'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9',
                   'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',
                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5',
                   'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',
                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36',
                   'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59',
                   'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81',
                   'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126',
                   'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151',
                   'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169',
                   'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189',
                   'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212',
                   'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227',
                   'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249',
                   'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',
                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268',
                   'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285',
                   'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313',
                   'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',
                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338',
                   'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',
                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17',
                   'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',
                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo',
                   'device_name', 'device_version', 'OS_id_30', 'version_id_30',
                   'browser_id_31', 'version_id_31', 'screen_width', 'screen_height', 'had_id']

cols_to_drop = [col for col in train.columns if col not in useful_features]
cols_to_drop.remove('isFraud')
cols_to_drop.remove('TransactionDT')

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

columns_a = ['TransactionAmt', 'id_02', 'D15']
columns_b = ['card1', 'card4', 'addr1']
for col_a in columns_a:
    for col_b in columns_b:
        for df in [train, test]:
            col_name_one = col_a + '_to_mean_' + col_b
            df[col_name_one] = df[col_a] / df.groupby([col_b])[col_a].transform('mean')
            col_name_two = col_a + '_to_std_' + col_b
            df[col_name_two] = df[col_a] / df.groupby([col_b])[col_a].transform('std')


# new feature - log of transaction amount
train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])
# New feature - decimal part of the transaction amount.
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

# New feature- day of week in which a transaction happened








# identity_data = pd.read_csv(identity_file)
# transaction_data = pd.read_csv(transaction_file)
# transaction_data = reduce_mem_usage(transaction_data)
# transaction_data = standard_scale(transaction_data, colname='TransactionDT', drop_col=True)
# transaction_data = standard_scale(transaction_data, colname='TransactionAmt', drop_col=True)
# transaction_data = one_hot_encoder(transaction_data, colname='ProductCD', drop_col=True)



# transaction_data = standard_scale(transaction_data, "TransactionAmt")
# print transaction_data.shape
# print transaction_data.head()

# transaction_data = one_hot_encoder(transaction_data, 'ProductCD')
# print transaction_data.shape
# print transaction_data.head(5)

# print transaction_data[transaction_data['isFraud'] == 1].shape
# unique = transaction_data['card6'].unique()
# print "shape is "
# print transaction_data[~transaction_data['card6'].isin(unique)].shape

# sns.boxplot(x="card4", y="C1", hue= "isFraud", data=transaction_data)
# plt.show()
