# -*- coding: utf-8 -*-
__author__ = 'plemberger'

import load_and_reshape
import train_and_predict
import data_viz



# load data from Kaggle files, reshape it and save it
load_and_reshape.load_Kaggle_files()
load_and_reshape.clean()
df_train, df_test = load_and_reshape.get_train_and_test()
#load_and_reshape.prepare_data_files_for_DD(df_train, df_test)
load_and_reshape.prepare_data_files_for_FL(df_train, df_test)

# un commentaire

# train various classifier and make predictions
train_and_predict.load_prepared_data_DD()
#train_and_predict.load_prepared_data_FL()
#train_and_predict.optimize_params()
train_and_predict.fit_model()
train_and_predict.predict_probabilities()
train_and_predict.write_submission_file()
train_and_predict.compute_feature_importance()



# construct graph and save them fo files
data_viz.load_prepared_data()
data_viz.construct_bar_plots()



# check wether different DD have FL in common. Answer is YES !!!
# conclusion FL alone are not reliable predictors

load_and_reshape.load_Kaggle_files()

df = load_and_reshape.df_train[['DepartmentDescription', 'FinelineNumber']]
groups_by_DD = df.groupby('DepartmentDescription')

DD_to_FL_set_mapping = {}
for dd, group in groups_by_DD:
    DD_to_FL_set_mapping[dd] = set(group['FinelineNumber'].unique().astype(int))

dd_list = pd.Series(load_and_reshape.df_train['DepartmentDescription'].unique()).dropna()
dd_list.sort(inplace = True)
nb_unique_DD = dd_list.size


found = False
for dd_1 in dd_list:
    set_1 = DD_to_FL_set_mapping[dd_1]
    for dd_2 in dd_list:
        if dd_list.tolist().index(dd_2) < dd_list.tolist().index(dd_1):
            set_2 = DD_to_FL_set_mapping[dd_2]
            intersection = set_1 & set_2
            if intersection:
                print('sets of FL associated with {} and {} have these elements in common: {}'.format(dd_1, dd_2, intersection))
                found = True

if not found:
    print('no intersection found')

