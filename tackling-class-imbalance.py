# -*- coding: utf-8 -*-
#@Author: saili
#@Time: 2019/12/16 19:42
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD



identity_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/train_identity.csv" #
transaction_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/train_transaction.csv" #(590540, 394)

test_identity_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/test_identity.csv"
test_transaction_file = "D:/Users/saili/Documents/kaggle/ieee-fraud-detection/test_transaction.csv"

train_transactions = pd.read_csv(transaction_file)
train_identity = pd.read_csv(identity_file)

print "Train data set is loaded"

train = train_transactions.merge(train_identity, how='left', left_index=True, right_index=True)
y_train = train['isFraud'].astype('unit8')
print 'Train shape ' + str(train.shape)

del train_transactions, train_identity
print 'Data set merged'

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


train = reduce_mem_usage(train)

X_train, X_test, y_train, y_test = train_test_split(train.drop('isFraud', axis=1), y_train, test_size=.2, random_state=1)


# Oversample minority class
X = pd.concat([X_train, y_train], axis = 1)
not_fraud = X[X.isFraud == 0]
fraud = X[X.isFraud == 1]

# upsample minority
fraud_upsampled = resample(fraud, replace=True, n_samples=len(not_fraud), random_state=27)

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
print upsampled.isFraud.value_counts()


y=upsampled.isFraud.value_counts()
sns.barplot(y=y,x=[0,1])
plt.title('upsampled data class count')
plt.ylabel('count')



# Undersample majority class
not_fraud_downsampled = resample(not_fraud, replace=False, n_samples=len(fraud), random_state=27)

# combine minority and downsample majority
downsampled  = pd.concat([not_fraud_downsampled, fraud])

# checking counts
downsampled.isFraud.value_counts()

y=downsampled.isFraud.value_counts()
sns.barplot(y=y,x=[0,1])
plt.title('downsampled data class count')
plt.ylabel('count')


from sklearn.datasets import make_classification
X, y = make_classification(n_classes=2, class_sep=1.5, weights=[0.9, 0.1], n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
df = pd.DataFrame(X)
df['target'] = y
df.target.value_counts().plot(kind='bar', title='Count Target')


def logistic(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    prob = lr.predict_proba(X_test)
    return (prob[:, 1], y_test)


probs, y_test = logistic(X, y)


def plot_pre_curve(y_test, probs):
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    plt.title('precision recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()



def plot_roc(y_test,prob):
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title("ROC curve")
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.show()


plot_pre_curve(y_test, probs)

plot_roc(y_test,probs)


#T-SNE Implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print "T-SNE took " + str(t1-t0)

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X)
t1 = time.time()
print 'PCA took ' + str(t1 - t0)

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X)
t1 = time.time()
print 'Truncated SVD took ' + str(t1 - t0)

# import imblearn

# Under-sampling: Tomek links

# Over-sampling: SMOTE
