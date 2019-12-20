import pandas as pd
import multiprocessing
from time import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import tree
import graphviz
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder




identity_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/train_identity.csv" #
transaction_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/train_transaction.csv" #(590540, 394)

test_identity_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/test_identity.csv"
test_transaction_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/test_transaction.csv"

sample_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/sample_submission.csv"

files = [test_identity_file, test_transaction_file,
         identity_file, transaction_file, sample_file]


def load_data(file):
    return pd.read_csv(file)


# if __name__=="__main__":

    # Loading all datasets using multiprocessing. This speads up a process a bit.
    # pool = multiprocessing.Pool(processes=4)
    # test_id, test_tr, train_id, train_tr, sub = pool.map(load_data, files)

    # merge data
    # train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
    # test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
    # del test_id, test_tr, train_id, train_tr


    # startdate = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')


test_id = pd.read_csv(test_identity_file)
test_tr = pd.read_csv(test_transaction_file)
train_id = pd.read_csv(identity_file)
train_tr = pd.read_csv(transaction_file)
sub = pd.read_csv(sample_file)


# merge data
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
del test_id, test_tr, train_id, train_tr
gc.collect()


def plot_numerical(feature):
    """
    Plot some information about a numerical feature for both and test set
    :param feature:
    :return:
    """
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
    sns.kdeplot(train[feature], ax=axes[0][0], label='Train')
    sns.kdeplot(test[feature], ax=axes[0][0], label='Test')

    sns.kdeplot(train[train['isFraud'] == 0][feature], ax=axes[0][1], label='isFraud: 0')
    sns.kdeplot(train[train['isFraud'] == 1][feature], ax=axes[0][1], label='isFraud: 1')

    test[feature].index += len(train)
    axes[1][0].plot(train[feature], '.', label='Train')
    axes[1][0].plot(test[feature], '.', label='Test')
    axes[1][0].set_xlabel('row index')
    axes[1][0].legend()
    test[feature].index -= len(train)

    axes[1][1].plot(train[train['isFraud'] == 0][feature], '.', label='isFraud 0')
    axes[1][1].plot(train[train['isFraud'] == 1][feature], '.', label='isFraud 1')
    axes[1][1].set_xlabel('row index')
    axes[1][1].legend()

    pd.DataFrame({'train': [train[feature].isnull().sum()], 'test': [test[feature].isnull().sum()]}).plot(
        kind='bar', rot=0, ax=axes[2][0]
    )
    pd.DataFrame({'isFraud 0': [train[(train['isFraud'] == 0) & (train[feature].isnull())][feature].shape[0]],
                  'isFraud 1': [train[(train['isFraud'] == 1) & (train[feature].isnull())][feature].shape[0]]})\
        .plot(kind='bar', rot=0, ax=axes[2][1])

    fig.suptitle(feature, fontsize=18)
    axes[0][0].set_title('Train/Test KDE distribution')
    axes[0][1].set_title('Target value KDE distribution')
    axes[1][0].set_title('Index versus value: Train/Test distribution')
    axes[1][1].set_title('Index versus value: Target distribution')
    axes[2][0].set_title('Number of NaNs');
    axes[2][1].set_title('Target value distribution among NaN values');


# This code is stolen from Chris Deotte.
def relax_data(df_train, df_test, col):
    cv1 = pd.DataFrame(df_train[col].value_counts().reset_index().rename({col: 'train'}, axis=1))
    cv2 = pd.DataFrame(df_test[col].value_counts().reset_index().rename({col: "test"}, axis=1))
    cv3 = pd.merge(cv1, cv2, on='index', how='outer')
    factor = len(df_test)/len(df_train)
    cv3['train'].fillna(0, inplace=True)
    cv3['test'].fillna(0, inplace=True)
    cv3['remove'] = False
    cv3['remove'] = cv3['remove'] | (cv3['train'] < len(df_train)/10000)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] < cv3['test']/3)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] > 3 * cv3['test'])
    cv3['new'] = cv3.apply(lambda x: x['index'] if x['remove'] is False else 0, axis=1)
    cv3['new'], _ = cv3['new'].factorize(sort=True)
    cv3.set_index('index', inplace=True)

    cc = cv3['new'].to_dict()
    df_train[col] = df_train[col].map(cc)
    df_test[col] = df_test[col].map(cc)
    return df_train, df_test


def plot_categorical(train, test, feature, target, values=5):
    """
    Plotting distribution for the selected amount of most frequent values between train and test along
    with distribution of target
    :param train:
    :param test:
    :param feature: name of the feature
    :param target: name of the target feature
    :param values: amount of most frequent values to look at
    :return:
    """
    df_train = pd.DataFrame(data={feature: train[feature], 'isTest': 0})
    df_test = pd.DataFrame(data={feature: test[feature], 'isTest': 1})
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df[df[feature].isin(df[feature].value_counts(dropna=False).head(values).index)]
    train = train[train[feature].isin(train[feature].value_counts(dropna=False).head(values).index)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    sns.countplot(data=df.fillna('NaN'), x=feature, hue='isTest', ax=axes[0])
    sns.countplot(data=train[[feature, target]].fillna('NaN'), x=feature, hue=target, ax=axes[1])
    axes[0].legend(['Train', 'Test'])


startdate = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
train['TransactionDT'] = train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
test['TransactionDT'] = test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

fig, axes = plt.subplots(1, 1, figsize=(16, 6))
# https://www.jianshu.com/p/061771f0afa9
train.set_index('TransactionDT').resample('D').mean()['isFraud'].plot(ax=axes).set_ylabel('isFraud mean', fontsize=14)
axes.set_title('Mean of isFraud by day', fontsize=16)


fig, axes = plt.subplots(1, 1, figsize=(16, 6))
train['TransactionDT'].dt.floor('d').value_counts().sort_index().plot(ax=axes).set_xlabel('Date', fontsize=14)
test['TransactionDT'].dt.floor('d').value_counts().sort_index().plot(ax=axes).\
    set_ylabel('Number of training examples', fontsize=14)
axes.set_title('Number of training examples by day', fontsize=16)
axes.legent(['Train', 'Test'])


# now combining both mean of isFraud by day and number of training examples by day into a single plot.
fig, ax1 = plt.subplots(figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['isFraud'].plot(ax=ax1, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylabel('isFraud mean', color='blue', fontsize=14)
ax2 = ax1.twinx()
train['TransactionDT'].dt.floor('d').value_counts().sort_index().plot(ax=ax2, color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.set_ylabel('Number of training examples', color='tab:orange', fontsize=14)
ax2.grid(False)


# feature: card1
y = train['isFraud']
X = pd.DataFrame()
X['card1'] = train['card1']
X['card1_count'] = train['card1'].map(
    pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False)
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47, stratify=y)
clf = DecisionTreeClassifier(max_leaf_nodes=4)
clf.fit(X_train, y_train)
print 'ROC AUC score: ', + str(roc_auc_score(y_test, clf.predict_proba(X_test)[0:1]))

# show the graph of the tree
tree_graph = tree.export_graphviz(clf, out_file=None, max_depth=10, impurity=False, feature_names=X.columns,
                                  class_names=['0', '1'], rounded=True, filled=True)
graphviz.Source(tree_graph)


plt.figure(figsize=(14, 6))
sns.kdeplot(X[y == 1]['card1'], label='isFraud 1')
sns.kdeplot(X[y == 0]['card1'], label='isFraud 0')
plt.plot([10881.5, 10881.5], [0.0000, 0.0001], sns.xkcd_rgb["black"], lw=2);
plt.plot([8750.0, 8750.0], [0.0000, 0.0001], sns.xkcd_rgb["red"], lw=2);


# train a boosting model on only one original feature card1
params = {'objective': 'binary',
          'boosting_type': 'gbdt',
          'subsample': 1,
          'bagging_seed': 11,
          'metric': 'auc',
          'random_state': 47}
X_train, X_test, y_train, y_test = train_test_split(X['card1'], y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier()
clf.fit(X_train.values.reshape(-1, 1), y_train)
print 'ROC AUC score ' + str(roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
print 'ROC AUC score: ' + str(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


plot_numerical('card1')


def covariate_shift(feature):
    df_card1_train = pd.DataFrame(data={feature: train[feature], 'isTest': 0})
    df_card1_test = pd.DataFrame(data={feature: test[feature], 'isTest': 1})

    # creating a single dataframe
    df = pd.concat([df_card1_train, df_card1_test], ignore_index=True)

    # Encoding if feature is categorical
    if str(df[feature].dtype) in ['object', 'category']:
        df[feature] = LabelEncoder().fit_transform(df[feature].astype(str))

    # Splitting it to a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df[feature], df['isTest'], test_size=0.33, random_state=47,
                                                        stratify=df['isTest'])
    clf = lgb.LGBMClassifier()
    clf.fit(X_train.values.reshape(-1, 1), y_train)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])
    del df, X_train, y_train, X_test, y_test
    gc.collect()

    return roc_auc

print 'Covariate Shift ROC AUC score: ' + str(covariate_shift('card1'))


# feature: ProductCD
plot_categorical(train, test, 'ProductCD', 'isFraud')

print 'Covariate shift ROC AUC: ' + str(covariate_shift('ProductCD'))


# feature: card2  理解不了

y = train['isFraud']
X = pd.DataFrame()
X['card2'] = train['card2']
X['card2_count'] = train['card2'].map(
    pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False)
)
result_df = pd.DataFrame()



test_X = pd.DataFrame()
test_X['card2'] = test['card2']
test_X['card2_count'] = test['card2'].map(
    pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False)
)

plot_numerical('card2')
print 'Covariate shift ROC AUC:' + str(covariate_shift('card2'))


# card4
df_train = pd.DataFrame(data={'card4': train['card4'], 'isTest': 0})
df_test = pd.DataFrame(data={'card4': test['card4'], 'isTest': 1})
df = pd.concat([df_train, df_test], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=df.fillna('NaN'), x='card4', hue='isTest', ax=axes[0])
sns.countplot(data=train[['card4', 'isFraud']].fillna('NaN'), x='card4', hue='isFraud', ax=axes[1])
axes[0].set_title('Train / Test distribution')
axes[1].set_title('Train distribution by isFraud')
axes[0].legend(['Train', 'Test'])

print 'Covariate shift ROC AUC:' + str(covariate_shift('card4'))


# card5
plot_numerical('card5')
print 'Covariate shift ROC AUC:' + str(covariate_shift('card5'))


# card6
df_train = pd.DataFrame(data={'card6': train['card6'], 'isTest': 0})
df_test = pd.DataFrame(data={'card6': test['card6'], 'isTest': 1})
df = pd.concat([df_train, df_test], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=df.fillna('NaN'), x='card6', hue='isTest', ax=axes[0])
sns.countplot(data=train[['card6', 'isFraud']].fillna('NaN'), x='card6', hue='isFraud', ax=axes[1])
axes[0].set_title('Train / Test distribution')
axes[1].set_title('Train distibution by isFraud')
axes[0].legend(['Train', 'Test'])

print 'Covariate shift ROC AUC:' + str(covariate_shift('card6'))


# addr1
y = train['isFraud']
X = pd.DataFrame()
X['addr1'] = train['addr1']
X['addr1_count'] = train['addr1'].map(
    pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False)
)
X['addr1'].fillna(0, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X['addr1'], y, test_size=0.33, random_state=47)
clf = DecisionTreeClassifier(max_leaf_nodes=4)
clf.fit(X_train.values.reshape(-1, 1), y_train)
print 'ROC AUC score:' + str(roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1]))

tree_graph = tree.export_graphviz(clf, out_file=None, max_depth=10, impurity=False, feature_names=['addr1'],
                                  class_names=['0', '1'], rounded=True, filled=True)
graphviz.Source(tree_graph)


plt.figure(figsize=(14, 6))
sns.kdeplot(X[y == 1]['addr1'], label='isFraud 1')
sns.kdeplot(X[y == 0]['addr1'], label='isFraud 0')
plt.plot([50, 50], [0, 0.008], sns.xkcd_rgb['black'], lw=2)


params = {'objective': 'binary',
          'boosting': 'gbdt',
          'subsample': 1,
          'bagging_seed': 11,
          'metric': 'auc',
          'random_state': 47}

X_train, X_test, y_train, y_test = train_test_split(X['addr1'], y, test_size=0.33,
                                                    random_state=47, stratify=y)
clf = lgb.LGBMClassifier()
clf.fit(X_train.values.reshape(-1, 1), y_train)
print 'ROC AUC score:' + str(roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1]))

plot_numerical('addr1')

print 'Covariate shift ROC AUC score:' + str(covariate_shift('addr1'))


# card1 to addr1 interaction
X = pd.DataFrame()
X['addr1'] = train['addr1']
X['card1'] = train['card1']
y = train['isFraud']
X['addr1'].fillna(0, inplace=True)
X['addr1_card1'] = X['addr1'].astype(str) + '_' + X['card1'].astype(str)
X['addr1_card1'] = LabelEncoder().fit_transform(X['addr1_card1'])


X_train, X_test, y_train, y_test = train_test_split(X[['addr1', 'card1']], y, test_size=0.33, random_state=47,
                                                    stratify=y)
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
print 'ROC AUC score:' + str(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

X_train, X_test, y_train, y_test = train_test_split(X[['addr1', 'card1', 'addr1_card1']], y, test_size=0.33, random_state=47,
                                                    stratify=y)
clf1 = lgb.LGBMClassifier()
clf1.fit(X_train, y_train)
print 'ROC AUC score:' + str(roc_auc_score(y_test, clf1.predict_proba(X_test)[:, 1]))


X['card1_count'] = train['card1'].map(
    pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False)
)
X['addr1_count'] = train['addr1_count'].map(
    pd.concat([train['addr1_count'], test['addr1_count']], ignore_index=True).value_counts(dropna=False)
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
print 'ROC AUC score:' + str(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


train['nulls'] = train.isnull().sum(axes=1)
test['nulls'] = test.isnull().sum(axis=1)
plot_numerical('nulls')
print 'Covariant shift ROC AUC:' + str(covariate_shift('nulls'))


# TransactionAmt and it's decimal part
plot_numerical('TransactionAmt')

fig, axes = plt.subplots(1, 1, figsize=(16, 6))
axes.set_title('Moving average of TransactionAmt', fontsize=16)
train[['TransactionDT', 'TransactionAmt']].set_index('TransactionDT').rolling(10000).mean().plot(ax=axes)
test[['TransactionDT', 'TransactionAmt']].set_index('TransactionDT').rolling(10000).mean().plot(ax=axes)
axes.legend(['Train', 'Test'])

fig, axes = plt.subplots(1, 1, figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['TransactionAmt'].plot(ax=axes)\
    .set_ylabel('TransactionAmt mean', fontsize=14)
test.set_index('TransactionDT').resample('D').mean()['TransactionAmt'].plot(ax=axes).\
    set_ylabel('TransactionAmt mean', fontsize=14)
axes.set_title('Mean of TransactionAmt by day', fontsize=16)

fig, ax1 = plt.subplots(figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['isFraud'].plot(ax=ax1, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylabel('isFraud mean by day', color='blue', fontsize=14)
ax2 = ax1.twinx()
train.set_index("TransactionDT").resample('D').mean()['TransactionAmt'].plot(ax=ax2, color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.set_ylabel('TransactionAmt mean by day', color='tab:orange', fontsize=14)
ax2.grid(False)


# Decimal part of transaction amount.
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int))*1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int))*1000).astype(int)
plot_numerical('TransactionAmt_decimal')

fig, axes = plt.subplots(1, 1, figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['TransactionAmt_decimal']\
    .plot(ax=axes).set_ylabel('TransactionAmt_decimal mean', fontsize=14)
test.set_index('TransactionDT').resample('D').mean()['TransactionAmt_decimal']\
    .plot(ax=axes).set_ylabel('TransactionAmt_decimal mean', fontsize=14)
axes.set_title('Mean of TransactionAmt_decimal by day', fontsize=16)


fig, ax1 = plt.subplots(figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['isFraud'].plot(ax=ax1, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylabel('isFraud mean by day', color='blue', fontsize=14)
ax2 = ax1.twinx()
train.set_index('TransactionDT').resample('D').mean()['TransactionAmt_decimal'].plot(ax=ax2, color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.set_ylabel('TransactionAmt_decimal mean by day', color='tab:orange', fontsize=14)
ax2.grid(False)


train['TransactionAmt_decimal_lenght'] = train['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str.len()
test['TransactionAmt_decimal_lenght'] = test['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str.len()


df_train = pd.DataFrame(data={'TransactionAmt_decimal_lenght': train['TransactionAmt_decimal_lenght'], 'isTest': 0})
df_test = pd.DataFrame(data={'TransactionAmt_decimal_lenght': test['TransactionAmt_decimal_lenght'], 'isTest': 1})
df = pd.concat([df_train, df_test], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=df.fillna('NaN'), x='TransactionAmt_decimal_lenght', hue='isTest', ax=axes[0])
sns.countplot(data=train[['TransactionAmt_decimal_lenght', 'isFraud']].fillna('NaN'), x='TransactionAmt_decimal_lenght',
              hue='isFraud', ax=axes[1])
axes[0].set_title('Train/ Test distribution')
axes[1].set_title('Train distribution by isFraud')
axes[0].legend(['Train', 'Test'])


print 'Covariant shift ROC AUC:' + str(covariate_shift('TransactionAmt'))
print 'Covariant shift ROC AUC:' + str(covariate_shift('TransactionAmt_decimal'))
print 'Covariant shift ROC AUC:' + str(covariate_shift('TransactionAmt_decimal_lenght'))


# v1
plot_numerical('V1')
print 'Covariate shift:' + str(covariate_shift('V1'))

# v2
plot_numerical('V2')
print 'Covariate shift:' + str(covariate_shift('V2'))


# data relaxation
train, test = relax_data(train, test, 'V258')
plot_numerical('V258')