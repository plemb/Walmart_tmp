# -*- coding: utf-8 -*-
__author__ = 'plemberger'


import load_and_reshape
import matplotlib.pyplot as plt
import pandas as pd


# path to image folder
image_folder = '/Users/pirminlemberger/PycharmProjects/Walmart/plots'


# global variables
X_train = None
y_train = None
X_test = None
X_and_y_train = None


# loading the prepared train and test data from existing files
def load_prepared_data():
    global X_train
    global X_test
    global y_train
    global X_and_y_train

    X_train = pd.read_csv(load_and_reshape.X_train_file_DD, index_col = 'VisitNumber')
    y_train = pd.read_csv(load_and_reshape.y_train_file, index_col = 'VisitNumber')['TripType']
    X_test = pd.read_csv(load_and_reshape.X_test_file_DD, index_col = 'VisitNumber')

    X_and_y_train = pd.concat([X_train, y_train], axis = 1)
    return


# compute product counts by TripType and by DeparmentDescription
def construct_bar_plots():
    cols = X_and_y_train.columns
    cols = cols.drop(['_Friday', '_Monday', '_Saturday',  '_Sunday', '_Thursday', '_Tuesday', '_Wednesday'])
    sum_by_TripType = X_and_y_train[cols].groupby('TripType').sum()

    # make one bar plot for the number of products by Department for each TripType
    plt.ioff()
    for  tripType in sum_by_TripType.index:
        trip = sum_by_TripType.ix[tripType]
        trip.sort_index(inplace = True)
        trip.plot(kind = 'bar', title = 'TripType ' + str(tripType))
        plt.savefig(image_folder + '/***TripType_' + str(tripType) + '.pdf')
        plt.close()
        print('--> bar plot for TripType {} saved'.format(tripType))
    plt.ion()
    return





