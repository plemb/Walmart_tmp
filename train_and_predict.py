# -*- coding: utf-8 -*-
__author__ = 'plemberger'


from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.externals import joblib

from load_and_reshape import *

import os


# global variables
clf = None
proba_predicted = None
y_predicted = None
X_train = None
X_test = None
y_train = None


# loading the prepared train and test data for DepartmentDescription features
def load_prepared_data_DD():
    global X_train
    global X_test
    global y_train

    X_train = pd.read_csv(X_train_file_DD, index_col ='VisitNumber')
    y_train = pd.read_csv(y_train_file, index_col = 'VisitNumber')['TripType']
    X_test = pd.read_csv(X_test_file_DD, index_col ='VisitNumber')

    return


# loading the prepared train and test data for FinelineNumber features
def load_prepared_data_FL():
    global X_train
    global X_test
    global y_train

    X_train = pd.read_csv(X_train_file_FL, index_col ='VisitNumber')
    print('--> loading X_train done!')

    y_train = pd.read_csv(y_train_file, index_col = 'VisitNumber')['TripType']
    print('--> loading y_train done!')

    X_test = pd.read_csv(X_test_file_FL, index_col ='VisitNumber')
    print('--> loading X_test done!')
    return


# create and fit a classifier
def fit_model():
    global clf
    print('--> fitting started')

    #clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    #clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0, verbose=2)
    #clf = LogisticRegression(verbose=2, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 500)
    clf = xgb.XGBClassifier(silent =  False)
    clf = clf.fit(X_train, y_train)
    print('--> fitting done!')

    joblib.dump(clf, '/Users/pirminlemberger/PycharmProjects/Walmart/classifiers/LR_lbfgs_with_FL.pkl')
    print('--> model saved!')

    return


# predict TripTypes values using the trained classifier
def predict_values():
    global y_predicted

    y_predicted = pd.Series(clf.predict(X_test))
    y_predicted.index = X_test.index
    if os.path.exists(y_predicted_file):
        os.remove(y_predicted_file)
    y_predicted.to_csv(y_predicted_file, header = ['TripType'])
    print('--> predicting y values done!')
    return


# predict TripTypes probabilities using the trained classifier
def predict_probabilities():
    global proba_predicted

    print('--> start computation of probabilities!')
    proba_predicted = pd.DataFrame(clf.predict_proba(X_test))
    proba_predicted.index = X_test.index
    print('--> predicting y probabilities done!')
    return


# write submission file according to Kaggle specified format
def write_submission_file():

    def create_header(n):
        return 'TripType_' + str(n)

    cols = pd.Series(clf.classes_).astype(int)
    cols = cols.map(create_header)
    proba_predicted.columns = cols

    if os.path.exists(proba_predicted_file):
        os.remove(proba_predicted_file)
    proba_predicted.to_csv(proba_predicted_file)
    print('--> submission file written!')
    return


# perform a cross-validation using the selected classifier
def cross_validate():
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='log_loss')
    print('--> cross validation done, score = {}'.format(-scores.mean()))


# compute features importance (available for RF and XGB)
def compute_feature_importance():
    # XGB
    xgb.plot_importance(clf)
    feature_importances = clf.booster().get_fscore()
    feature_importances = pd.Series(feature_importances).sort_values()

    # RF and other classifiers
    #feature_names = X_train.columns
    #importances = clf.feature_importances_
    #feature_importances = pd.Series(data=importances, index=feature_names)
    #feature_importances = feature_importances.sort_values()

    print('--> feature importances:')
    print(feature_importances)
    return


# use a grid search CV to optimize hyper-parameters of selected classifier
def optimize_params():
    global clf

    # XGB
    parameters_to_try = [{'learning_rate':0.1, 'n_estimators':100}, {'learning_rate':0.2, 'n_estimators':50}]
    clf = xgb.XGBClassifier(seed=0)

    # RF
    #parameters_to_try = {'n_estimators': [250, 500], 'max_depth': [10, None]}
    #clf = RandomForestClassifier(random_state=0, verbose=2)

    grid_clf = GridSearchCV(clf, parameters_to_try, cv=3, scoring='log_loss')
    grid_clf.fit(X_train, y_train)

    print('--> cross validation done!')

    print('--> best parameters:')
    print(clf.best_params_)
    print()

    print('--> scores for all combinations of parameters:')
    print(clf.grid_scores_)
    print()
    return









