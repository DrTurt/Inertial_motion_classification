import pandas as pd
import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer_names, accuracy_score, recall_score, precision_score, \
    precision_recall_curve, roc_auc_score, f1_score
from classification import *


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
                      'kernel': ["linear", "poly", "rbf", "sigmoid"],
                      'degree': [2, 3, 4, 5],
                      'gamma': ["scale", "auto"],
                      'tol': [0.01, 0.001, 0.0001],
                      'decision_function_shape': ["ovo", "ovr"]
                      }
    print("break")


def set_up_for_multi_layer_perceptron():
    mlp = multi_layer_perceptron()
    all_parameters = mlp.get_params()
    all_scorers = get_scorer_names()
    scorers = ['accuracy', 'recall_micro', 'precision_micro', 'f1_micro', 'roc_auc_ovr']
    parameter_dict = {'hidden_layer_sizes': [(10,), (50,), (100,), (200,), (500,),
                                             (10, 6), (50, 6), (100, 6), (200, 6), (500, 6),
                                             (10, 21), (50, 21), (100, 21), (200, 21), (500, 21),
                                             (10, 50), (50, 50), (100, 50), (200, 50), (500, 50),
                                             (10, 100), (50, 100), (100, 100), (200, 100), (500, 100),
                                             (10, 100, 6), (50, 100, 6), (100, 100, 6), (200, 100, 6),
                                             (10, 100, 21), (50, 100, 21), (100, 100, 21), (200, 100, 21)],
                      'activation': ["identity", "logistic", "tanh", "relu"],
                      'solver': ["lbfgs", "sgd", "adam"],
                      'alpha': [0.00001, 0.0001, 0.0005, 0.001],
                      'learning_rate': ["constant", "invscaling", "adaptive"],
                      'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01],
                      'max_iter': [100, 200, 500, 1000],
                      'shuffle': [True, False],
                      'tol': [0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                      'momentum': [0.7, 0.8, 0.9]
                      }
    print("break")


def run_grid_search(dataset, classifier, parameters, scorers):
    data, target = split_to_data_and_target(dataset)
    grid_searcher = GridSearchCV(estimator=classifier, param_grid=parameters, scoring=scorers, refit='accuracy')
    grid_searcher.fit(data, target)
    return grid_searcher
