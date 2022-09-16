import pandas as pd
import numpy as np
import math
from time import time
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer_names, accuracy_score, recall_score, precision_score, \
    precision_recall_curve, roc_auc_score, f1_score
from classification import *
from logger import *

log_this = custom_logger("grid search log")


def set_up_for_decision_tree():
    dt = decision_tree()
    all_parameters = dt.get_params()
    all_scorers = get_scorer_names()
    scorers = ['accuracy', 'recall_micro', 'precision_micro', 'f1_micro', 'roc_auc_ovr']
    parameter_dict = {'ccp_alpha': [0.0, 0.01, 0.05, 0.1],
                      'criterion': ["gini", "entropy", "log_loss"],
                      'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],
                      'splitter': ["random", "best"]
                      }
    return dt, scorers, parameter_dict


def set_up_for_random_forest():
    rf = random_forest()
    all_parameters = rf.get_params()
    all_scorers = get_scorer_names()
    scorers = ['accuracy', 'recall_micro', 'precision_micro', 'f1_micro', 'roc_auc_ovr']
    parameter_dict = {'n_estimators': [10, 50, 100, 200, 500],
                      'criterion': ["gini", "log_loss", "entropy"],
                      'min_impurity_decrease': [0.0, 0.001, 0.01, 0.05, 0.1],
                      'ccp_alpha': [0.0, 0.001, 0.01, 0.05, 0.1],
                      }
    return rf, scorers, parameter_dict


def set_up_for_support_vector_machine():
    svm = support_vector_machine()
    all_parameters = svm.get_params()
    all_scorers = get_scorer_names()
    scorers = ['accuracy', 'recall_micro', 'precision_micro', 'f1_micro', 'roc_auc_ovr']
    parameter_dict = {'C': [0.5, 0.8, 1.0, 1.2, 1.5],
                      'kernel': ["poly", "rbf", "sigmoid"],
                      'degree': [2, 3, 4, 5],
                      'gamma': ["scale", "auto"],
                      'tol': [0.01, 0.001, 0.0001],
                      'decision_function_shape': ["ovo", "ovr"]
                      }
    return svm, scorers, parameter_dict


def set_up_for_multi_layer_perceptron():
    mlp = multi_layer_perceptron()
    all_parameters = mlp.get_params()
    all_scorers = get_scorer_names()
    scorers = ['accuracy', 'recall_micro', 'precision_micro', 'f1_micro', 'roc_auc_ovr']
    parameter_dict = {'hidden_layer_sizes': [(50,), (100,), (200,), (500,), (1000,)],
                      'alpha': [0.0001, 0.0005, 0.001],
                      'learning_rate': ["constant", "invscaling", "adaptive"],
                      'learning_rate_init': [0.0005, 0.001, 0.005, 0.01],
                      'max_iter': [1000, 5000, 10000]
                      }
    return mlp, scorers, parameter_dict


def run_grid_search(dataset, classifier, parameters, scorers):
    tock = time()
    data, target = split_to_data_and_target(dataset)
    grid_searcher = GridSearchCV(estimator=classifier,
                                 param_grid=parameters,
                                 scoring=scorers,
                                 refit='accuracy',
                                 n_jobs=2,
                                 verbose=1)
    grid_searcher.fit(data, target)
    tick = time()
    elapsed_time = tick - tock
    elapsed_seconds = int(elapsed_time % 60)
    elapsed_minutes = int((elapsed_time % 3600) / 60)
    elapsed_hours = int(elapsed_time / 3600)
    log_this.info("The total elapsed time for running all datasets was {} hours {} minutes and {} seconds"
                  .format(elapsed_hours, elapsed_minutes, elapsed_seconds))
    return grid_searcher
