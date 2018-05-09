# coding: utf-8

from __future__ import division
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import gc
import warnings

warnings.filterwarnings('ignore')


root_path = '../'  # '/media/xiaoxy/2018-Kaggle-AdTrackingFraud/'


def corr(first_file_path, second_file_path):
    # Assuming first column is "class_name_id"
    first_df = pd.read_csv(first_file_path, index_col=0)
    second_df = pd.read_csv(second_file_path, index_col=0)
    class_names = ['is_attributed']

    for class_name in class_names:
        # All correlations
        print('Class: %s' % class_name)
        print('Pearson\'s correlation score: %0.6f' % first_df[class_name].corr(second_df[class_name], method='pearson'))
        print('Kendall\'s correlation score: %0.6f' % first_df[class_name].corr(second_df[class_name], method='kendall'))
        print('Spearman\'s correlation score: %0.6f' % first_df[class_name].corr(second_df[class_name], method='spearman'))
        ks_stat, p_value = ks_2samp(first_df[class_name].values, second_df[class_name].values)
        print('Kolmogorov-Smirnov test: KS-stat=%.6f p-value=%.3e' % (ks_stat, p_value))


###################################### Cal correlation ######################################

file1_path = root_path + 'data/output/sub/20180507-lgb-0.981609(r2100).csv'
file2_path = root_path + 'data/output/sub/20180506-0.99146(r2000).csv'

# Cal correlation between 2 result file
corr(file1_path, file2_path)

###################################### Blending ######################################

test_files = [file1_path, file2_path]
weights = [0.4, 0.6]
column_name = 'is_attributed'

model_test_data = []
for test_file in test_files:
    print('Read ' + test_file)
    model_test_data.append(pd.read_csv(test_file, encoding='utf-8'))
n_models = len(model_test_data)

print('Blending...')
test_predict_column = [0.] * len(model_test_data[0][column_name])
for ind in range(0, n_models):
    test_predict_column += model_test_data[ind][column_name] * weights[ind]
print('Blend done!')

print('Save result...')
final_result = model_test_data[0]['click_id']
final_result = pd.concat((final_result, pd.DataFrame(
    {column_name: test_predict_column})), axis=1)
final_result.to_csv(root_path + 'data/output/sub/blend_201805081151.csv', index=False)
print('Save result done!')
