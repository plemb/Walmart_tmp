# -*- coding: utf-8 -*-
__author__ = 'plemberger'

import numpy as np
import pandas as pd
import os



# TripType              - a categorical id representing the type of shopping trip the customer made. This is the ground truth that you are predicting. TripType_999 is an "other" category.
# VisitNumber           - an id corresponding to a single trip by a single customer
# Weekday               - the weekday of the trip
# Upc                   - the UPC number of the product purchased
# ScanCount             - the number of the given item that was purchased. A negative value indicates a product return.
# DepartmentDescription - a high-level description of the item's department
# FinelineNumber        - a more refined category for each of the products, created by Walmart



# global variables
df_train = None
df_test = None


# limitting the number of samples for testing
head_only = False
nb_samples = 100

# log the reshaping process to the console
verbose = False


# path to various train and test sets
kaggle_train_file    = '/Users/pirminlemberger/PycharmProjects/Walmart/data/train.csv'
X_train_file_DD      = '/Users/pirminlemberger/PycharmProjects/Walmart/data/X_train_DD.csv'
X_train_file_FL      = '/Users/pirminlemberger/PycharmProjects/Walmart/data/X_train_FL.csv'
y_train_file         = '/Users/pirminlemberger/PycharmProjects/Walmart/data/y_train.csv'

kaggle_test_file     = '/Users/pirminlemberger/PycharmProjects/Walmart/data/test.csv'
X_test_file_DD       = '/Users/pirminlemberger/PycharmProjects/Walmart/data/X_test_DD.csv'
X_test_file_FL       = '/Users/pirminlemberger/PycharmProjects/Walmart/data/X_test_FL.csv'
y_predicted_file     = '/Users/pirminlemberger/PycharmProjects/Walmart/data/***_y_predicted.csv'
proba_predicted_file = '/Users/pirminlemberger/PycharmProjects/Walmart/data/***_proba_predicted.csv'


# load original kaggle train and test files
def load_Kaggle_files():
    global df_train
    global df_test

    if head_only:
        df_train = pd.read_csv(kaggle_train_file).head(nb_samples)
        df_test = pd.read_csv(kaggle_test_file).head(nb_samples)
    else:
        df_train = pd.read_csv(kaggle_train_file)
        df_test = pd.read_csv(kaggle_test_file)
    return


# access original kaggle train and test files
def get_train_and_test():
    return df_train, df_test


# dropping duplicates in train set
def clean():
    global df_train

    nb_rows_before = df_train.shape[0]
    df_train =  df_train.drop_duplicates()
    nb_rows_after = df_train.shape[0]
    nb_dropped = nb_rows_before - nb_rows_after
    print('--> dropped {} lines'.format(nb_dropped))

    # replacing negative values in ScanCount by 0
    def remneg(x):
        if x < 0:
            return 0
        else:
            return x

    df_train['ScanCount'] = df_train['ScanCount'].map(remneg)
    df_test['ScanCount'] = df_test['ScanCount'].map(remneg)

    return


# reshape data to create a DataFrame with a single row for each VisitNumber (using DD feature)
# Uses a fully vetoctorized procedure and dummyfication for best performance. This method does not allow however
# reshaping the data with the FinelineNumber feature because too much memory is used
def reshape_for_DD(df):
    # replace missing DepartmentDescription by 'NaN' so we can groupby properly
    df['DepartmentDescription'].replace(np.NaN, '*NaN*', inplace = True)

    # group by VisitNumber and DepartmentDescription
    grouped = df.drop(['Upc','FinelineNumber'], axis=1).groupby(['VisitNumber','DepartmentDescription'])

    tmp = grouped.sum().drop('TripType', axis =1)
    tmp = tmp.reset_index(level = 'DepartmentDescription')

    data_by_VisitNumber = grouped.first().reset_index(level = 'VisitNumber').groupby('VisitNumber').first()
    wd_by_VisitNumber = data_by_VisitNumber['Weekday']
    tt_by_VisitNumber = data_by_VisitNumber['TripType']

    # dummify DepartmentDescription values
    dummies = pd.get_dummies(data = tmp, prefix = '', columns = ['DepartmentDescription'])

    # reshape dummies using broadcasting
    sc = dummies['ScanCount']
    dd = dummies[dummies.columns.drop(['ScanCount'])]
    reshaped = pd.DataFrame(sc.values.reshape(sc.size,1) * dd.values)
    reshaped.columns = dd.columns
    reshaped['VisitNumber'] = tmp.index

    # flattening vertically to keep one single row for each VisitNumber
    reshaped = reshaped.groupby('VisitNumber').sum()
    reshaped['Weekday'] = wd_by_VisitNumber.values
    reshaped['TripType'] = tt_by_VisitNumber.values

    # dummify Weekday values
    reshaped = pd.get_dummies(data = reshaped, prefix = '', columns = ['Weekday'])

    # drop column corresponding to NaN values
    reshaped.drop('_*NaN*', axis = 1, inplace = True)

    return reshaped


# reshape data to create a DataFrame with a single row for each VisitNumber (using FL feature)
# Uses an explicit loop on VisitNumbers to avoid the dummification procedure used in reshape_using_DepartmentDescription
# which uses too much memory
def reshape_for_FL(df):
    df.drop(['DepartmentDescription', 'Upc'], axis = 1, inplace = True)
    df['FinelineNumber'].replace(np.NaN, -999, inplace = True)
    cols_to_keep = df.columns.drop(['TripType', 'Weekday'])
    groups1 = df[cols_to_keep].groupby(['VisitNumber','FinelineNumber'])
    groups2 = df.groupby(['VisitNumber','FinelineNumber'])

    result1 = groups1.sum()
    result2 = groups2.first()
    result2['ScanCount'] = result1['ScanCount']
    result3 = result2.reset_index('FinelineNumber')

    fl_and_sc = result3[['FinelineNumber','ScanCount']]
    fl_and_sc_by_VisitNumber = fl_and_sc.reset_index('VisitNumber').groupby('VisitNumber')
    tt_and_wd_by_VisitNumber = result3[['TripType','Weekday']].reset_index('VisitNumber').groupby('VisitNumber').first()
    tt_and_wd_by_VisitNumber = pd.get_dummies(tt_and_wd_by_VisitNumber, prefix = '', columns = ['Weekday'])

    reshaped_data = []

    nb_visit = df['VisitNumber'].unique().size
    print('--> total number of visits: {}'.format(nb_visit))
    counter = 0

    for vn, visit_df in fl_and_sc_by_VisitNumber:
        counter = counter + 1
        if counter % 5000 == 0:
            print('--> {} visit numbers added'.format(counter))

        sc = visit_df['ScanCount']
        fl = visit_df['FinelineNumber'].values.astype(int).astype(str)
        vn_row = pd.Series(data = sc.values, index = fl)
        reshaped_data.append(vn_row)
    print('--> reshaping step 1 done!')

    visit_numbers = tt_and_wd_by_VisitNumber.index
    reshaped = pd.DataFrame(reshaped_data, index = visit_numbers)
    print('--> reshaping step 2 done!')

    reshaped.replace(np.NaN, 0, inplace = True)
    reshaped = pd.concat([reshaped, tt_and_wd_by_VisitNumber], axis = 1)
    print('--> reshaping step 3 done!')

    return reshaped


# build X_train, X_test and y_train for predictions using DepartmentDescriptions and save to files
def prepare_data_files_for_DD(df1, df2):
    df = pd.concat([df1, df2])
    df_reshaped = reshape_for_DD(df)

    df_train_reshaped = df_reshaped[np.bitwise_not(np.isnan(df_reshaped['TripType']))]
    print('--> reshaping of train set done!')

    df_test_reshaped = df_reshaped[np.isnan(df_reshaped['TripType'])]
    print('--> reshaping of test set done!')

    y_train = df_train_reshaped['TripType']
    if os.path.exists(y_train_file):
        os.remove(y_train_file)
    y_train.to_csv(y_train_file, header='TripType')

    X_train = df_train_reshaped.drop('TripType', axis = 1)
    if os.path.exists(X_train_file_DD):
        os.remove(X_train_file_DD)
    X_train.to_csv(X_train_file_DD)
    print('--> reshaped train set writtent to file')

    X_test = df_test_reshaped.drop('TripType', axis = 1)
    if os.path.exists(X_test_file_DD):
        os.remove(X_test_file_DD)
    X_test.to_csv(X_test_file_DD)
    print('--> reshaped test set writtent to file')



# build X_train, X_test and y_train for predictions using FinelineNumber and save to files
def prepare_data_files_for_FL(df1, df2):
    df = pd.concat([df1, df2])
    df_reshaped = reshape_for_FL(df)

    df_train_reshaped = df_reshaped[np.bitwise_not(np.isnan(df_reshaped['TripType']))]
    print('--> reshaping of train set done!')

    df_test_reshaped = df_reshaped[np.isnan(df_reshaped['TripType'])]
    print('--> reshaping of test set done!')

    y_train = df_train_reshaped['TripType']
    if os.path.exists(y_train_file):
        os.remove(y_train_file)
    y_train.to_csv(y_train_file, header='TripType')

    X_train = df_train_reshaped.drop('TripType', axis = 1)
    if os.path.exists(X_train_file_FL):
        os.remove(X_train_file_FL)
    X_train.to_csv(X_train_file_FL)
    print('--> reshaped train set writtent to file')

    X_test = df_test_reshaped.drop('TripType', axis = 1)
    if os.path.exists(X_test_file_FL):
        os.remove(X_test_file_FL)
    X_test.to_csv(X_test_file_FL)
    print('--> reshaped test set writtent to file')



# create a small typical DataFrame with NaN's to be used for debugging
def create_sample_df():
    df_train = pd.read_csv(kaggle_train_file)
    df_25 = df_train[df_train['VisitNumber'] == 25].drop('TripType', axis = 1)
    df_26 = df_train[df_train['VisitNumber'] == 26]
    df_28 = df_train[df_train['VisitNumber'] == 28]

    df_26.set_value(77, 'DepartmentDescription', np.NaN)
    df_26.set_value(76, 'FinelineNumber', None)
    df = pd.concat([df_25, df_26, df_28], axis=0)
    df.drop('Upc', axis = 1)
    return df


# show the first groups by VisitNumber
def display_sample_visits():
    df_train = pd.read_csv(kaggle_train_file).head(100)
    groups = df_train.groupby('VisitNumber')
    for visit_number, group in groups:
        print('-----------------')
        print('--> visit n°: {}'.format(visit_number))
        print('-----------------')
        print(group)
        print()







