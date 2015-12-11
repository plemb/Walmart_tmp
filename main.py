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
