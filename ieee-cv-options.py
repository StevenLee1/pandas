# -*- coding: utf-8 -*-
#@Author: saili
#@Time: 2019/12/19 20:36

import random
import numpy as np
import datetime
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score



def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


SEED = 42
seed_everything(SEED)
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
print START_DATE



lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_jobs': -1,
    'learning_rate': 0.01,
    'num_leaves': 2**8,
    'max_depth': -1,
    'tree_learner': 'serial',
    'colsample_bytree': 0.7,
    'subsample_freq': 1,
    'subsample': 0.7,
    'n_estimators': 20000,
    'max_bin': 255,
    'verbose': -1,
    'seed': SEED,
    'early_stopping_rounds': 100
}

train_df = pd.DataFrame()
test_df = pd.DataFrame()

train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)* 12 + train_df['DT_M'].dt.month

test_df = train_df[train_df['DT_M'] == train_df['DT_M'].max()].reset_index(drop=True)
train_df = train_df[train_df['DT_M'] < (train_df['DT_M'].max())].reset_index(drop=True)

########################### Encode Str columns
# For all such columns (probably not)
# we already did frequency encoding (numeric feature)
# so we will use astype('category') here


# 除了占用内存节省外，另一个额外的好处是计算效率有了很大的提升。
# 因为对于Category类型的Series，str字符的操作发生在.cat.categories的非重复值上，而并非原Series上的所有元素上。
# 也就是说对于每个非重复值都只做一次操作，然后再向与非重复值同类的值映射过去。

# 但是Category数据的使用不是很灵活。例如，插入一个之前没有的值，首先需要将这个值添加到.categories的容器中，然后再添加值。

for col in list(train_df):
    if train_df[col].dtype == 'O':
        print col
        train_df[col] = train_df[col].fillna('unseen_before_label')
        test_df[col] = test_df[col].fillna('unseen_before_label')

        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        le = LabelEncoder()
        le.fit(list(train_df[col]) + list(test_df[col]))
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')


########################### Model Features
# Remove Some Features
rm_cols = [
    'TransactionID','TransactionDT', # These columns are pure noise right now
    TARGET,                          # Not target in features))
    'DT_M'                           # Column that we used to simulate test set
]

# Remove V columns (for faster training)
rm_cols += ['V'+str(i) for i in range(1,340)]

# Final features
features_columns = [col for col in list(train_df) if col not in rm_cols]





RESULTS = test_df[['TransactionID', TARGET]]

X, y = train_df[features_columns], train_df[TARGET]
P = test_df[features_columns]

for n_rounds in [500, 1000, 2500, 5000]:
    print "#"*20
    print 'No Validation training...' + str(n_rounds) + 'boosting rounds'
    corrected_lgb_params = lgb_params.copy()
    corrected_lgb_params['n_estimators'] = n_rounds
    corrected_lgb_params['early_stopping_rounds'] = None

    train_data = lgb.Dataset(X, label=y)
    estimator = lgb.train(corrected_lgb_params, train_data)

    RESULTS['no_validation_' + str(n_rounds)] = estimator.predict(P)
    print 'AUC score' + str(roc_auc_score(RESULTS[TARGET], RESULTS['no_validation_' + str(n_rounds)]))
    print '#' * 20



# CV Concept
# Main strategy¶
#
#     1. Divide Train set in subsets (Training set itself + Validation set)
#     2. Define Validation Metric (in our case it is ROC-AUC)
#     3. Stop training when Validation metric stops improving
#     4. Make predictions for Test set

N_SPLITS = 3
print '#' * 20
print 'KFold training'

# You can find oof name for this strategy
# oof - Out Of Fold
# as we will use one fold as validation
# and stop training when validation metric
# stops improve

from sklearn.model_selection import KFold
folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Main data
X, y = train_df[features_columns], train_df[TARGET]
# Test data
P = test_df[features_columns]
RESULTS['kfold'] = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print 'Fold: ' + str(fold_ + 1)
    tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
    vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=vl_y)

    estimator = lgb.train(lgb_params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000)
    RESULTS['kfold'] = estimator.predict(P)

print 'AUC score ' + str(roc_auc_score(RESULTS[TARGET], RESULTS['kfold']))
print '#'*20


## We have two "problems" here
## 1st: Training score goes upto 1 and it's not normal situation
## It's nomally means that model did perfect or
## almost perfect match between "data fingerprint" and target
## we definitely should stop before to generalize better
## 2nd: Our LB probing gave 0.936 and it is too far away from validation score
## some difference is normal, but such gap is too big


# StratifiedKFold
# There are situations when normal kfold split doesn't perform well because of train set imbalance.
# We can use StratifiedKFold to garant that each split will have same number of positives and negatives samples.

print "#"*20
print 'StratifiedKFold training...'

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Main Data
X, y = train_df[features_columns], train_df[TARGET]
# Test Data and export DF
P = test_df[features_columns]
RESULTS['stratifiedkfold'] = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=y)):
    print 'Fold:' + str(fold_+1)
    tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
    vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=vl_y)

    estimator = lgb.train(lgb_params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000)

    RESULTS['stratifiedkfold'] += estimator.predict(P)/N_SPLITS

print 'AUC score' + str(roc_auc_score(RESULTS[TARGET], RESULTS['stratifiedkfold']))
print '#' * 20

# bagging_fraction':0.8           ###  数据采样

print '#'*20
print 'LBO training'

train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month

main_train_set = train_df[train_df['DT_M'] < (train_df['DT_M'].max())].reset_index(drop=True)
validation_set = train_df[train_df['DT_M'] == train_df['DT_M'].max()].reset_index(drop=True)

folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
# Main data
X, y = main_train_set[features_columns], main_train_set[TARGET]
# Validation data
v_X, v_y = validation_set[features_columns], validation_set[TARGET]

estimators_bestround = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print 'Fold: ' + str(fold_ + 1)
    tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(v_X, label=v_y)

    estimator = lgb.train(lgb_params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000)
    estimators_bestround.append(estimator.current_iteration())

## Now we have "mean Best round" and we can train model on full set
corrected_lgb_params = lgb_params.copy()
corrected_lgb_params['n_estimators'] = int(np.mean(estimators_bestround))
corrected_lgb_params['early_stopping_rounds'] = None
print '#'*20
print 'Mean Best round: ' + str(corrected_lgb_params['n_estimators'])

X, y = train_df[features_columns], train_df[TARGET]

P = test_df[features_columns]
RESULTS['lbo'] = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print 'Fold: ' + str(fold_ + 1)
    tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
    train_data = lgb.Dataset(tr_x, tr_y)

    estimator = lgb.train(corrected_lgb_params, train_data)

    RESULTS['lbo'] += estimator.predict(P)/N_SPLITS

print 'AUC score' + roc_auc_score(RESULTS[TARGET], RESULTS['lbo'])
print '#'*20




# kfold does not work well
train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month

main_train_set = train_df[train_df['DT_M'] < (train_df['DT_M'].max())].reset_index(drop=True)
validation_set = train_df[train_df['DT_M'] == train_df['DT_M'].max()].reset_index(drop=True)

# Main Data
X,y = main_train_set[features_columns], main_train_set[TARGET]
# Validation Data
v_X, v_y = validation_set[features_columns], validation_set[TARGET]
estimators_bestround = []

for current_model in range(3):
    print 'Model: ' + str(current_model + 1)
    SEED += 1
    seed_everything(SEED)
    corrected_lgb_params = lgb_params.copy()
    corrected_lgb_params['seed'] = SEED

    train_data = lgb.Dataset(X, label=y)
    valid_data = lgb.Dataset(v_X, label=v_y)

    estimator = lgb.train(
        corrected_lgb_params,
        train_data,
        valid_sets=[train_data, valid_data],
        verbose_eval=1000,
    )

    estimators_bestround.append(estimator.current_iteration())



corrected_lgb_params = lgb_params.copy()
corrected_lgb_params['n_estimators'] = int(np.mean(estimators_bestround))
corrected_lgb_params['early_stopping_rounds'] = None
print '#'*10
print 'Mean Best round:' + str(corrected_lgb_params['n_estimators'])
# Main Data
X,y = train_df[features_columns], train_df[TARGET]

# Test Data
P = test_df[features_columns]
RESULTS['lbo_full'] = 0
NUMBER_OF_MODELS = 3

for current_model in range(NUMBER_OF_MODELS):
    print 'Model: ' + str(current_model +1)
    SEED += 1
    seed_everything(SEED)
    train_data = lgb.Dataset(X, label=y)

    estimator = lgb.train(
        corrected_lgb_params,
        train_data
    )

    RESULTS['lbo_full'] += estimator.predict(P) / NUMBER_OF_MODELS

print 'AUC score' + str(roc_auc_score(RESULTS[TARGET], RESULTS['lbo_full']))
print '#'*20








