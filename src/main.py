# coding: utf-8

from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import KFold
import xgboost as xgb
import lightgbm as lgb
import cPickle
import time
import datetime
import math
import gc
import warnings

warnings.filterwarnings('ignore')


root_path = '../'  # '/media/xiaoxy/2018-Kaggle-AdTrackingFraud/'
predictors = []


########################################### Helper function ###########################################

def encode_onehot(df, column_name):
    df_onehot = pd.get_dummies(df[column_name], prefix=column_name)
    df_all = pd.concat([df.drop([column_name], axis=1), df_onehot], axis=1)
    predictors.append(column_name)
    return df_all


def encode_count(df, column_name):
    le = preprocessing.LabelEncoder()
    le.fit(list(df[column_name].values))
    df[column_name] = le.transform(list(df[column_name].values))
    predictors.append(column_name)
    return df


def merge_count(df, columns_groupby, new_column_name, type='uint64'):
    add = pd.DataFrame(df.groupby(columns_groupby).size()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_nunique(df, columns_groupby, column, new_column_name, type='uint64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].nunique()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_cumcount(df, columns_groupby, column, new_column_name, type='uint64'):
    df[new_column_name] = df.groupby(columns_groupby)[column].cumcount().values.astype(type)
    predictors.append(new_column_name)
    return df


def merge_median(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].median()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_mean(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].mean()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_sum(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].sum()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    # predictors.append(new_column_name)  # bug: twice
    return df


def merge_max(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].max()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_min(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].min()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_std(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].std()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_var(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].var()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_rank(df, columns_groupby, column, new_column_name, ascending=True, type='uint64'):
    df[new_column_name] = df.groupby(columns_groupby)[column].rank(ascending=ascending)
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df


def merge_feat_count(df, df_feat, columns_groupby, column, new_column_name=""):
    df_count = pd.DataFrame(df_feat.groupby(columns_groupby)[column].count()).reset_index()
    if not new_column_name:
        df_count.columns = columns_groupby + [column + "_gb_%s_count" % ("_".join(columns_groupby))]
    else:
        df_count.columns = columns_groupby + [new_column_name]
    df = df.merge(df_count, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_count.columns[-1])
    return df


def merge_feat_nunique(df, df_feat, columns_groupby, column, new_column_name=""):
    df_nunique = pd.DataFrame(df_feat.groupby(columns_groupby)[column].nunique()).reset_index()
    if not new_column_name:
        df_nunique.columns = columns_groupby + [column + "_%s_nunique" % ("_".join(columns_groupby))]
    else:
        df_nunique.columns = columns_groupby + [new_column_name]
    df = df.merge(df_nunique, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_nunique.columns[-1])
    return df


def merge_feat_mean(df, df_feat, columns_groupby, column, new_column_name=""):
    df_mean = pd.DataFrame(df_feat.groupby(columns_groupby)[column].mean()).reset_index()
    if not new_column_name:
        df_mean.columns = columns_groupby + [column + "_%s_mean" % ("_".join(columns_groupby))]
    else:
        df_mean.columns = columns_groupby + [new_column_name]
    df = df.merge(df_mean, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_mean.columns[-1])
    return df


def merge_feat_std(df, df_feat, columns_groupby, column, new_column_name=""):
    df_std = pd.DataFrame(df_feat.groupby(columns_groupby)[column].std()).reset_index()
    if not new_column_name:
        df_std.columns = columns_groupby + [column + "_%s_std" % ("_".join(columns_groupby))]
    else:
        df_std.columns = columns_groupby + [new_column_name]
    df = df.merge(df_std, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_std.columns[-1])
    return df


def merge_feat_median(df, df_feat, columns_groupby, column, new_column_name=""):
    df_median = pd.DataFrame(df_feat.groupby(columns_groupby)[column].median()).reset_index()
    if not new_column_name:
        df_median.columns = columns_groupby + [column + "_%s_median" % ("_".join(columns_groupby))]
    else:
        df_median.columns = columns_groupby + [new_column_name]
    df = df.merge(df_median, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_median.columns[-1])
    return df


def merge_feat_max(df, df_feat, columns_groupby, column, new_column_name=""):
    df_max = pd.DataFrame(df_feat.groupby(columns_groupby)[column].max()).reset_index()
    if not new_column_name:
        df_max.columns = columns_groupby + [column + "_%s_max" % ("_".join(columns_groupby))]
    else:
        df_max.columns = columns_groupby + [new_column_name]
    df = df.merge(df_max, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_max.columns[-1])
    return df


def merge_feat_min(df, df_feat, columns_groupby, column, new_column_name=""):
    df_min = pd.DataFrame(df_feat.groupby(columns_groupby)[column].min()).reset_index()
    if not new_column_name:
        df_min.columns = columns_groupby + [column + "_%s_min" % ("_".join(columns_groupby))]
    else:
        df_min.columns = columns_groupby + [new_column_name]
    df = df.merge(df_min, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_min.columns[-1])
    return df


def merge_feat_sum(df, df_feat, columns_groupby, column, new_column_name=""):
    df_sum = pd.DataFrame(df_feat.groupby(columns_groupby)[column].sum()).reset_index()
    if not new_column_name:
        df_sum.columns = columns_groupby + [column + "_%s_sum" % ("_".join(columns_groupby))]
    else:
        df_sum.columns = columns_groupby + [new_column_name]
    df = df.merge(df_sum, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_sum.columns[-1])
    return df


def merge_feat_var(df, df_feat, columns_groupby, column, new_column_name=""):
    df_var = pd.DataFrame(df_feat.groupby(columns_groupby)[column].var()).reset_index()
    if not new_column_name:
        df_var.columns = columns_groupby + [column + "_%s_var" % ("_".join(columns_groupby))]
    else:
        df_var.columns = columns_groupby + [new_column_name]
    df = df.merge(df_var, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_var.columns[-1])
    return df


def merge_feat_quantile(df, df_feat, columns_groupby, column, quantile_n, new_column_name=""):
    df_quantile = pd.DataFrame(df_feat.groupby(columns_groupby)[column].quantile(quantile_n)).reset_index()
    if not new_column_name:
        df_quantile.columns = columns_groupby + [column + "_%s_quantile" % ("_".join(columns_groupby))]
    else:
        df_quantile.columns = columns_groupby + [new_column_name]
    df = df.merge(df_quantile, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_quantile.columns[-1])
    return df


def merge_feat_skew(df, df_feat, columns_groupby, column, new_column_name=""):
    df_skew = pd.DataFrame(df_feat.groupby(columns_groupby)[column].skew()).reset_index()
    if not new_column_name:
        df_skew.columns = columns_groupby + [column + "_%s_skew" % ("_".join(columns_groupby))]
    else:
        df_skew.columns = columns_groupby + [new_column_name]
    df = df.merge(df_skew, on=columns_groupby, how="left").fillna(0)
    predictors.append(df_skew.columns[-1])
    return df


def merge_rank_sp(df, feat1, feat2, ascending):
    df.sort_values([feat1, feat2], inplace=True, ascending=ascending)
    df['rank'] = range(df.shape[0])
    min_rank = df.groupby(feat1, as_index=False)['rank'].agg({'min_rank': 'min'})
    df = pd.merge(df, min_rank, on=feat1, how='left')
    df['rank'] = df['rank'] - df['min_rank']
    predictors.append('rank')
    del df['min_rank']
    return df


def log(info):
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info)


def log_shape(train, test):
    log('Train data shape: %s' % str(train.shape))
    log('Test data shape: %s' % str(test.shape))


def process_date(df):
    format = '%Y-%m-%d %H:%M:%S'
    df['date'] = pd.to_datetime(df['click_time'], format=format)
    df['month'] = df['date'].dt.month.astype('uint8')
    df['weekday'] = df['date'].dt.weekday.astype('uint8')
    df['day'] = df['date'].dt.day.astype('uint8')
    df['hour'] = df['date'].dt.hour.astype('uint8')
    df['minute'] = df['date'].dt.minute.astype('uint8')
    df['second'] = df['date'].dt.second.astype('uint8')
    df['tm_hour'] = (df['hour'] + df['minute'] / 60.0).astype('float32')
    df['tm_hour_sin'] = (df['tm_hour'].map(lambda x: math.sin((x - 12) / 24 * 2 * math.pi))).astype('float32')
    df['tm_hour_cos'] = (df['tm_hour'].map(lambda x: math.cos((x - 12) / 24 * 2 * math.pi))).astype('float32')
    del df['click_time']
    return df


########### Construct features function - begin ###########

def cal_next_time_delta(df, suffix, type='float32'):
    groupby_columns = [
        {'columns': ['ip', 'app', 'channel', 'device', 'os']},
        {'columns': ['ip', 'os', 'device']},
        {'columns': ['ip', 'os', 'device', 'app']}
    ]
    # Calculate the time to next click for each group
    for spec in groupby_columns:
        # Name of new feature
        new_name = '{}_{}'.format('_'.join(spec['columns']), suffix)
        # Unique list of features to select
        all_features = spec['columns'] + ['date']
        # Run calculation
        log('Calculate ' + suffix + '...')
        df[new_name] = (df[all_features].groupby(spec['columns']).date.shift(-1) - df.date).dt.seconds.astype(type)
        predictors.append(new_name)
        gc.collect()
    return df


def cal_prev_time_delta(df, suffix, type='float32'):
    groupby_columns = [
        {'columns': ['ip', 'channel']},
        {'columns': ['ip', 'os']}
    ]
    # Calculate the time to prev click for each group
    for spec in groupby_columns:
        # Name of new feature
        new_name = '{}_{}'.format('_'.join(spec['columns']), suffix)
        # Unique list of features to select
        all_features = spec['columns'] + ['date']
        # Run calculation
        log('Calculate ' + suffix + '...')
        df[new_name] = (df.date - df[all_features].groupby(spec['columns']).date.shift(+1)).dt.seconds.astype(type)
        predictors.append(new_name)
        gc.collect()
    return df


def cal_cvr(train, test, type='float32'):
    train['cvr_gb_ip_day_hour'] = 0
    train['cvr_gb_ip_app'] = 0
    train['cvr_gb_ip_app_os'] = 0

    # Define group by list
    idh = ['ip', 'day', 'hour']
    ia = ['ip', 'app']
    iao = ['ip', 'app', 'os']

    kf = KFold(train.shape[0], n_folds=5, shuffle=True, random_state=7)

    for i, (train_index, test_index) in enumerate(kf):
        log('Fold ' + str(i) + ' begin...')

        # Divide train/test fold
        tr = train.iloc[train_index, :train.shape[1] - 3]
        te = train.iloc[test_index, :train.shape[1] - 3]

        # Calculate sum of label of train folds
        log('Cal sum_label_gb_ip_day_hour')
        tr = merge_sum(tr, idh, 'is_attributed', 'sum_label_gb_ip_day_hour')
        log('Cal sum_label_gb_ip_app')
        tr = merge_sum(tr, ia, 'is_attributed', 'sum_label_gb_ip_app')
        log('Cal sum_label_gb_ip_app_os')
        tr = merge_sum(tr, iao, 'is_attributed', 'sum_label_gb_ip_app_os')

        # Calculate cvr of train folds with using smothing technique
        tr['cvr_gb_ip_day_hour'] = GaussianSmoth().update_moment(tr['count_gb_ip_day_hour'], tr['sum_label_gb_ip_day_hour'])
        tr['cvr_gb_ip_app'] = GaussianSmoth().update_moment(tr['count_gb_ip_app'], tr['sum_label_gb_ip_app'])
        tr['cvr_gb_ip_app_os'] = GaussianSmoth().update_moment(tr['count_gb_ip_app_os'], tr['sum_label_gb_ip_app_os'])

        # Merge test fold with cvr features of train folds
        te = te.merge(tr[['cvr_gb_ip_day_hour'] + idh].drop_duplicates(subset=idh, keep='first'), on=idh, how='left')
        te = te.merge(tr[['cvr_gb_ip_app'] + ia].drop_duplicates(subset=ia, keep='first'), on=ia, how='left')
        te = te.merge(tr[['cvr_gb_ip_app_os'] + iao].drop_duplicates(subset=iao, keep='first'), on=iao, how='left')

        # Put it in train
        train['cvr_gb_ip_day_hour'] += te['cvr_gb_ip_day_hour']
        train['cvr_gb_ip_app'] += te['cvr_gb_ip_app']
        train['cvr_gb_ip_app_os'] += te['cvr_gb_ip_app_os']

        del tr, te
        log('Fold ' + str(i) + ' Done!')

    # Convert type
    train['cvr_gb_ip_day_hour'] = train['cvr_gb_ip_day_hour'].astype(type)
    train['cvr_gb_ip_app'] = train['cvr_gb_ip_app'].astype(type)
    train['cvr_gb_ip_app_os'] = train['cvr_gb_ip_app_os'].astype(type)

    # Merge cvr of train to test
    test = test.merge(train[['cvr_gb_ip_day_hour'] + idh].drop_duplicates(subset=idh, keep='first'), on=idh, how='left')
    test = test.merge(train[['cvr_gb_ip_app'] + ia].drop_duplicates(subset=ia, keep='first'), on=ia, how='left')
    test = test.merge(train[['cvr_gb_ip_app_os'] + iao].drop_duplicates(subset=iao, keep='first'), on=iao, how='left')

    predictors.append('cvr_gb_ip_day_hour')
    predictors.append('cvr_gb_ip_app')
    predictors.append('cvr_gb_ip_app_os')

    return train, test


########### Construct features function - end ###########

def spilt_local_train_test(df, train_size, test_size):
    local_train = df[:train_size]
    local_test = df[train_size:train_size + test_size]
    return local_train, local_test


def get_model_input_data(train, test, is_local):
    feat = ['ip', 'app', 'device', 'os', 'channel', 'hour']
    for f in feat:
        if f not in predictors:
            predictors.append(f)
    train_x = train[predictors]
    train_y = train.is_attributed.values
    if is_local == 1:
        test_x = test[train_x.columns.values]
        test_y = test.is_attributed.values
        return train_x, train_y, test_x, test_y
    else:
        test_x = test[train_x.columns.values]
        return train_x, train_y, test_x


def lgb_cv(train_feature, train_label, test_feature, test_label, params, folds, rounds):
    start = time.clock()
    print(train_feature.columns)
    params['scale_pos_weight'] = float(len(train_label[train_label == 0])) / len(train_label[train_label == 1])
    dtrain = lgb.Dataset(train_feature, label=train_label, categorical_feature=['app', 'device', 'os', 'channel', 'hour'])
    dtest = lgb.Dataset(test_feature, label=test_label, categorical_feature=['app', 'device', 'os', 'channel', 'hour'])
    num_round = rounds
    print('LightGBM run cv: ' + 'round: ' + str(rounds))
    res = lgb.train(params, dtrain, num_round, valid_sets=[dtest], valid_names=['test'], verbose_eval=1, early_stopping_rounds=20)
    elapsed = (time.clock() - start)
    print('Time used:', elapsed, 's')
    return res.best_iteration, res.best_score['test']['auc'], res


def lgb_predict(train_feature, train_label, test_feature, rounds, params):
    dtrain = lgb.Dataset(train_feature, label=train_label, categorical_feature=['app', 'device', 'os', 'channel', 'hour'])
    num_round = rounds
    model = lgb.train(params, dtrain, num_round, valid_sets=[dtrain], verbose_eval=1)
    predict = model.predict(test_feature)
    return model, predict


def store_result(test_index, pred, name):
    result = pd.DataFrame({'click_id': test_index, 'is_attributed': pred})
    result.to_csv(root_path + 'data/output/sub/' + name + '.csv', index=False, sep=',')
    return result


class GaussianSmoth(object):
    def __init__(self, alpha=0, beta=0):
        self.alpha = alpha
        self.beta = beta

    def update_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        self.alpha = (mean + 0.000001) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        self.beta = (1.000001 - mean) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        print self.alpha, self.beta
        return (self.alpha + success) / (self.alpha + self.beta + tries)

    def __compute_moment(self, tries, success):
        # Cal mean and variance
        '''moment estimation'''
        ctr_list = []
        mean = (success / tries).mean()
        if len(tries) == 1:
            var = 0
        else:
            var = (success / tries).var()
        return mean, var


########################################### Read data ###########################################

log('Read data...')
dtypes = {
    'click_id': 'uint32',
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8'
}
train = pd.read_csv(root_path + 'data/input/train.csv', header=0, sep=',', dtype=dtypes, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
test_supplement = pd.read_csv(root_path + 'data/input/test_supplement.csv', header=0, sep=',', dtype=dtypes, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time'])
gc.collect()
log('Read data done!')
log_shape(train, test_supplement)

########################################### Preprocess ###########################################

log('Process date...')
train = process_date(train)
test_supplement = process_date(test_supplement)
gc.collect()
log('Process date done!')
log_shape(train, test_supplement)

########################################### Feature engineer ###########################################

train_len = len(train)
log('Train size:' + str(train_len))

log('Train append test_supplement...')
df = train.append(test_supplement).reset_index(drop=True)
del train
del test_supplement
gc.collect()
log('Train append test_supplement done!')

log('Before feature engineer')
log('Num of features: ' + str(len(df.columns)))
log('Features: ' + str(df.columns))

# Construct features...
log('Cal next_time_delta')
df = cal_next_time_delta(df, 'next_time_delta', 'float32')
gc.collect()
log('Cal prev_time_delta')
df = cal_prev_time_delta(df, 'prev_time_delta', 'float32')
gc.collect()
log('Cal nunique_channel_gb_ip')
df = merge_nunique(df, ['ip'], 'channel', 'nunique_channel_gb_ip', 'uint32')
gc.collect()
log('Cal nunique_app_gb_ip_device_os')
df = merge_nunique(df, ['ip', 'device', 'os'], 'app', 'nunique_app_gb_ip_device_os', 'uint32')
gc.collect()
log('Cal nunique_hour_gb_ip_day')
df = merge_nunique(df, ['ip', 'day'], 'hour', 'nunique_hour_gb_ip_day', 'uint32')
gc.collect()
log('Cal nunique_app_gb_ip')
df = merge_nunique(df, ['ip'], 'app', 'nunique_app_gb_ip', 'uint32')
gc.collect()
log('Cal nunique_os_gb_ip_app')
df = merge_nunique(df, ['ip', 'app'], 'os', 'nunique_os_gb_ip_app', 'uint32')
gc.collect()
log('Cal nunique_device_gb_ip')
df = merge_nunique(df, ['ip'], 'device', 'nunique_device_gb_ip', 'uint32')
gc.collect()
log('Cal nunique_channel_gb_app')
df = merge_nunique(df, ['app'], 'channel', 'nunique_channel_gb_app', 'uint32')
gc.collect()
log('Cal cumcount_os_gb_ip')
df = merge_cumcount(df, ['ip'], 'os', 'cumcount_os_gb_ip', 'uint32');
gc.collect()
log('Cal cumcount_app_gb_ip_device_os')
df = merge_cumcount(df, ['ip', 'device', 'os'], 'app', 'cumcount_app_gb_ip_device_os', 'uint32');
gc.collect()
log('Cal count_gb_ip_day_hour')
df = merge_count(df, ['ip', 'day', 'hour'], 'count_gb_ip_day_hour', 'uint32');
gc.collect()
log('Cal count_gb_ip_app')
df = merge_count(df, ['ip', 'app'], 'count_gb_ip_app', 'uint32');
gc.collect()
log('Cal count_gb_ip_app_os')
df = merge_count(df, ['ip', 'app', 'os'], 'count_gb_ip_app_os', 'uint32');
gc.collect()
log('Cal var_day_gb_ip_app_os')
df = merge_var(df, ['ip', 'app', 'os'], 'day', 'var_day_gb_ip_app_os', 'float32')
gc.collect()
# Construct features done!

log('After feature engineer')
log('Num of features: ' + str(len(df.columns)))
log('Features: ' + str(df.columns))

########### All features save & reload - begin ###########

# Save all features
cPickle.dump(df, open(root_path + 'data/output/feat/all.p', 'wb'))

# # Reload all features
# df = cPickle.load(open(root_path + 'data/output/feat/all.p', 'rb'))
# train_len = 184903891
# dtypes = {
#     'click_id': 'uint32',
#     'ip': 'uint32',
#     'app': 'uint16',
#     'device': 'uint16',
#     'os': 'uint16',
#     'channel': 'uint16',
#     'is_attributed': 'uint8'
# }
# predictors = ['ip', 'app', 'device', 'os', 'channel', 'hour',
#               'next_time_delta', 'prev_time_delta',
#               'nunique_channel_gb_ip', 'nunique_app_gb_ip_device_os',
#               'nunique_hour_gb_ip_day', 'nunique_app_gb_ip', 'nunique_os_gb_ip_app',
#               'nunique_device_gb_ip', 'nunique_channel_gb_app',
#               'cumcount_os_gb_ip', 'cumcount_app_gb_ip_device_os',
#               'count_gb_ip_day_hour', 'count_gb_ip_app', 'count_gb_ip_app_os',
#               'var_day_gb_ip_app_os']

########### All features save & reload - end ###########

log('Train test_supplement divid...')
train = df[:train_len]
test_supplement = df[train_len:]
del df
gc.collect()
log_shape(train, test_supplement)
log('Train test_supplement divid done!')

log('Read test...')
test = pd.read_csv(root_path + 'data/input/test.csv', header=0, sep=',', dtype=dtypes, usecols=['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time'], parse_dates=['click_time'])
log('Test data original shape: ' + str(test.shape))

test = test.merge(test_supplement.drop_duplicates(subset=['ip', 'app', 'device', 'os', 'channel', 'date'], keep='first'), left_on=['ip', 'app', 'device', 'os', 'channel', 'click_time'], right_on=['ip', 'app', 'device', 'os', 'channel', 'date'], how='left')
test.drop(['click_time'], axis=1, inplace=True)
del test_supplement
gc.collect()
log_shape(train, test)
log('Read test done!')

# Cal cvr features
log('Cal cvr...')
train, test = cal_cvr(train, test, 'float32')
log('Cal cvr done!')

########### CVR features save & reload - begin ###########

cvr_feats = ['cvr_gb_ip_day_hour', 'cvr_gb_ip_app', 'cvr_gb_ip_app_os']

# Save cvr features
cPickle.dump(train[cvr_feats], open(root_path + 'data/output/feat/train_cvr.p', 'wb'))
cPickle.dump(test[cvr_feats], open(root_path + 'data/output/feat/test_cvr.p', 'wb'))

# # Reload cvr features
# train_cvr = cPickle.load(open(root_path + 'data/output/feat/train_cvr.p', 'rb'))
# test_cvr = cPickle.load(open(root_path + 'data/output/feat/test_cvr.p', 'rb'))
# train = pd.concat([train, train_cvr], axis=1)
# test_cvr = pd.concat([test, test_cvr], axis=1)
# del train_cvr, test_cvr

########### CVR features save & reload - end ###########

########################################### Split dataset for local ###########################################

log('Split dataset to get local train/test set...')
local_train_size = 10000000  # 182403890
local_test_size = 2500000
local_train, local_test = spilt_local_train_test(train, local_train_size, local_test_size)
log('Split dataset to get local train/test set done!')

log('================================= Local data info =====================================')
log('Local train shape:' + str(local_train.shape))
log('Local test shape:' + str(local_test.shape))
log('Local train label ratio (0-1):' + str(local_train.is_attributed.value_counts().values * 1.0 / local_train.shape[0]))
log('Local train label number (0-1):' + str(local_train.is_attributed.value_counts().values))
log('Local train min/max date:' + str(local_train.date.min()) + ',' + str(local_train.date.max()))
log('Local test min/max date:' + str(local_test.date.min()) + ',' + str(local_test.date.max()))
log('=======================================================================================')

log('================================= Online data info =====================================')
log('Online train shape:' + str(train.shape))
log('Online test shape:' + str(test.shape))
log('Online train label ratio (0-1):' + str(train.is_attributed.value_counts().values * 1.0 / train.shape[0]))
log('Online train label number (0-1):' + str(train.is_attributed.value_counts().values))
log('Online train min/max date:' + str(train.date.min()) + ',' + str(train.date.max()))
log('Online train min/max date:' + str(test.date.min()) + ',' + str(test.date.max()))
log('=======================================================================================')

log('Get local model input data...')
local_train_x, local_train_y, local_test_x, local_test_y = get_model_input_data(local_train, local_test, is_local=1)
del local_train
del local_test
gc.collect()
log_shape(local_train_x, local_test_x)
log('Get local model input data done!')

log('Get online model input data...')
online_train_x, online_train_y, online_test_x = get_model_input_data(train, test, is_local=0)
del train
del test
gc.collect()
log_shape(online_train_x, online_test_x)
log('Get online model input data done!')

########################################### Model ###########################################

########################################### LigthGBM ###########################################

config_lgb = {
    'rounds': 10000,
    'folds': 5
}

params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'xentropy',
    'metric': 'auc',
    'learning_rate': 0.02,
    # 'is_unbalance': 'true',  # Because training data is unbalance (replaced with scale_pos_weight)
    'scale_pos_weight': 200,  # Because training data is extremely unbalanced
    'num_leaves': 31,  # We should let it be smaller than 2^(max_depth)
    'max_depth': -1,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 128,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # Frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0.99,  # L1 regularization term on weights
    'reg_lambda': 0.9,  # L2 regularization term on weights
    'nthread': 24,
    'verbose': 1,
    'seed': 8
}

iterations_lgb, best_score_lgb, model_cv_lgb = lgb_cv(local_train_x, local_train_y, local_test_x, local_test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])

# pred_lgb = model_cv_lgb.predict(online_test_x)

model_lgb, pred_lgb = lgb_predict(online_train_x, online_train_y, online_test_x, iterations_lgb, params_lgb)

importance_lgb = sorted(zip(online_train_x.columns, model_cv_lgb.feature_importance("gain")), key=lambda x: x[1], reverse=True)
importance_lgb = pd.DataFrame({'feature': importance_lgb})
importance_lgb = importance_lgb.apply(lambda x: pd.Series(x['feature']), axis=1)
importance_lgb.columns = ['feature', 'importance']
importance_lgb.to_csv(root_path + 'data/output/feat_imp/importance-lgb-20180507-%f(r%d).csv' % (best_score_lgb, iterations_lgb), index=False)

res_lgb = store_result(pd.read_csv(root_path + 'data/input/test.csv', header=0, sep=',', usecols=['click_id']).click_id.astype(int), pred_lgb, '20180507-lgb-%f(r%d)' % (best_score_lgb, iterations_lgb))

########### Model save and reload - begin ###########

# Save model
log('Save model...')
model_lgb.save_model(root_path + 'data/output/model/lgb-%f(r%d).txt' % (best_score_lgb, iterations_lgb))
log('Model best score:' + str(best_score_lgb))
log('Model best iteration:' + str(iterations_lgb))
log('Save model done!')

# # Reload model
# model_lgb = lgb.Booster(model_file=root_path + 'data/output/model/lgb-0.981609(r2100).txt')

########### Model save and reload - end ###########
