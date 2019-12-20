# -*- coding: utf-8 -*-


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


X, y = make_classification(n_classes=2, class_sep=1.5, weights=[0.9, 0.1], n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
params = {
          'objective': 'binary',
          'max_depth': -1,
          "boosting_type": "gbdt",
          "metric": 'auc',
          'random_state': 47,
          'learning_rate': [0.001, 0.005, 0.01]
         }



# Step1. 学习率和估计器及其数目
# 我们先把学习率先定一个较高的值，这里取 learning_rate = 0.1
# 其次确定估计器boosting/boost/boosting_type的类型，不过默认都会选gbdt
# 确定估计器的数目，也就是boosting迭代的次数，也可以说是残差树的数目，参数名为n_estimators/num_iterations/num_round/num_boost_round

params = {
          'objective': 'binary',
          "boosting_type": "gbdt",
          "metric": 'auc',
          'learning_rate': 0.1,
          'max_depth': 6,     # 根据问题来定咯，由于我的数据集不是很大，所以选择了一个适中的值，其实4-10都无所谓。
          'num_leaves': 50,   # 由于lightGBM是leaves_wise生长，官方说法是要小于2^max_depth
          'bagging_fraction': 0.8,  # subsample, 数据采样
          'feature_fraction': 0.8  # colsample_bytree, 特征采样
         }


train_data = lgb.Dataset(data=X_train, label=y_train)
test_data = lgb.Dataset(data=X_test, label=y_test)
# bst = lgb.train(params, train_data, 100, valid_sets=[test_data], early_stopping_rounds=10)

cv_results = lgb.cv(params=params, train_set=train_data, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
       early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
print cv_results
print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', cv_results['auc-mean'][-1])

# 由于我的数据集不是很大，所以在学习率为0.1时，最优的迭代次数只有43。
# 那么现在，我们就代入(0.1, 43)进入其他参数的tuning。但是还是建议，在硬件条件允许的条件下，学习率还是越小越好。


# Step2. max_depth 和 num_leaves
from sklearn.model_selection import GridSearchCV
### 我们可以创建lgb的sklearn模型，使用上面选择的(学习率，评估器数目)
model_lgb = lgb.LGBMClassifier(objective='binary', num_leaves=50, learning_rate=0.1, n_estimators=20, max_depth=6,
                               metric='auc', bagging_fraction=0.8, feature_fraction=0.8)
params_test1 = {
    'max_depth': range(3, 8, 2),
    'num_leaves': range(50, 170, 30)
}

gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring="roc_auc", cv=5, verbose=1, n_jobs=4)
gsearch1.fit(X_train, y_train)
print gsearch1.best_score_
print gsearch1.best_params_
print gsearch1.best_estimator_
print gsearch1.best_index_
# 我们将我们这步得到的最优解代入第三步。
# 其实，我这里只进行了粗调，如果要得到更好的效果，可以将max_depth在7附近多取几个值，num_leaves在80附近多取几个值。
# 千万不要怕麻烦，虽然这确实很麻烦。




# Step3: min_data_in_leaf 和 min_sum_hessian_in_leaf
# 说到这里，就该降低过拟合了。

# min_data_in_leaf 是一个很重要的参数, 也叫min_child_samples，它的值取决于训练数据的样本个树和num_leaves.
# 将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合。

# min_sum_hessian_in_leaf：也叫min_child_weight，使一个结点分裂的最小海森值之和，
# 真拗口（Minimum sum of hessians in one leaf to allow a split. Higher values potentially decrease overfitting）
params_test3 = {
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight': [0.001, 0.002]
}
model_lgb = lgb.LGBMClassifier(objective='binary', num_leaves=50, learning_rate=0.1, n_estimators=20, max_depth=3,
                               metric='auc', bagging_fraction=0.8, feature_fraction=0.8)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(X_train, y_train)
print gsearch3.best_score_
print gsearch3.best_params_
print gsearch3.best_estimator_
print gsearch3.best_index_



# Step4: feature_fraction 和 bagging_fraction
# 这两个参数都是为了降低过拟合的。
# feature_fraction参数来进行特征的子抽样。这个参数可以用来防止过拟合及提高训练速度。

# bagging_fraction+bagging_freq参数必须同时设置，
# bagging_fraction相当于subsample样本采样，可以使bagging更快的运行，同时也可以降拟合。
# bagging_freq默认0，表示bagging的频率，0意味着没有使用bagging，k意味着每k轮迭代进行一次bagging。

params_test4 = {
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
}
model_lgb = lgb.LGBMClassifier(objective='binary', num_leaves=50, learning_rate=0.1, n_estimators=20, max_depth=3,
                               metric='auc', bagging_freq=5, min_child_samples=19, min_child_weight=0.001)
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(X_train, y_train)
print gsearch4.best_score_
print gsearch4.best_params_
print gsearch4.best_estimator_
print gsearch4.best_index_



# Step5: 正则化参数
# 正则化参数lambda_l1(reg_alpha), lambda_l2(reg_lambda)，毫无疑问，是降低过拟合的，两者分别对应l1正则化和l2正则化。我们也来尝试一下使用这两个参数
params_test6={
    'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
}
model_lgb = lgb.LGBMClassifier(objective='binary', num_leaves=50, learning_rate=0.1, n_estimators=20, max_depth=3,
                               metric='auc', bagging_freq=5, min_child_samples=19, min_child_weight=0.001, feature_fraction=0.8)
gsearch5 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
gsearch5.fit(X_train, y_train)
print gsearch5.best_score_
print gsearch5.best_params_
print gsearch5.best_estimator_
print gsearch5.best_index_



# step6: 降低learning_rate
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',

    'learning_rate': 0.005,
    'num_leaves': 50,
    'max_depth': 3,
    'min_data_in_leaf': 19,

    'subsample': 1,
    'colsample_bytree': 0.8,
}
cv_results = lgb.cv(
    params, train_data, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='auc',
    early_stopping_rounds=50, verbose_eval=100, show_stdv=True)
print cv_results
print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', cv_results['auc-mean'][-1])