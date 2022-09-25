import pandas as pd
import numpy as np
from logger import *
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

log_this = custom_logger("classification log")


def split_to_data_and_target(data_frame):
    data = data_frame.drop(columns=["target"])
    target = data_frame["target"]
    return data, target


def naive_bayes():
    nb = GaussianNB()
    return nb


def decision_tree():
    dt = DecisionTreeClassifier()
    return dt


def random_forest():
    rf = RandomForestClassifier()
    return rf


def support_vector_machine():
    svm = SVC()
    return svm


def multi_layer_perceptron():
    mlp = MLPClassifier()
    return mlp


def flat_run_all_classifiers(dataset, dataset_name):
    X, y = split_to_data_and_target(dataset)
    nb = naive_bayes()
    dt = decision_tree()
    rf = random_forest()
    svm = support_vector_machine()
    mlp = multi_layer_perceptron()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    nb.fit(X_train, y_train)
    score = nb.score(X_test, y_test)
    log_this.info("Naive Bayes classifier accuracy on {} dataset: {}".format(dataset_name, score))

    dt.fit(X_train, y_train)
    score = dt.score(X_test, y_test)
    log_this.info("Decision tree classifier accuracy on {} dataset: {}".format(dataset_name, score))

    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    log_this.info("Random forest classifier accuracy on {} dataset: {}".format(dataset_name, score))

    svm.fit(X_train, y_train)
    score = svm.score(X_test, y_test)
    log_this.info("Support vector machine classifier accuracy on {} dataset: {}".format(dataset_name, score))

    mlp.fit(X_train, y_train)
    score = mlp.score(X_test, y_test)
    log_this.info("Multi-layer perceptron classifier accuracy on {} dataset: {}".format(dataset_name, score))


def run_optimised_classifiers(dataset, dataset_name, param_dict):
    X, y = split_to_data_and_target(dataset)
    nb = naive_bayes()
    key_name = "decision_tree_{}".format(dataset_name)
    dt_param_dict = param_dict[key_name]
    dt = DecisionTreeClassifier(ccp_alpha=dt_param_dict['ccp_alpha'],
                                criterion=dt_param_dict['criterion'],
                                min_impurity_decrease=dt_param_dict['min_impurity_decrease'],
                                splitter=dt_param_dict['splitter']
                                )
    key_name = "random_forest_{}".format(dataset_name)
    rf_param_dict = param_dict[key_name]
    rf = RandomForestClassifier(ccp_alpha=rf_param_dict['ccp_alpha'],
                                criterion=rf_param_dict['criterion'],
                                min_impurity_decrease=rf_param_dict['min_impurity_decrease'],
                                n_estimators=rf_param_dict['n_estimators'], n_jobs=4)
    key_name = "support_vector_machine_{}".format(dataset_name)
    svm_param_dict = param_dict[key_name]
    svm = SVC(C=svm_param_dict['C'],
              decision_function_shape=svm_param_dict['decision_function_shape'],
              degree=svm_param_dict['degree'],
              gamma=svm_param_dict['gamma'],
              kernel=svm_param_dict['kernel'],
              probability=svm_param_dict['probability'],
              tol=svm_param_dict['tol'])
    key_name = "multi_layer_perceptron_{}".format(dataset_name)
    mlp_param_dict = param_dict[key_name]
    mlp = MLPClassifier(alpha=mlp_param_dict['alpha'],
                        hidden_layer_sizes=eval(mlp_param_dict['hidden_layer_sizes']),
                        learning_rate=mlp_param_dict['learning_rate'],
                        learning_rate_init=mlp_param_dict['learning_rate_init'],
                        max_iter=mlp_param_dict['max_iter'])

    log_this.info("Running Naive Bayes cross validation on {} dataset".format(dataset_name))
    nb_cv_results = cross_validate(nb, X, y, cv=5, n_jobs=4,
                                   scoring=['accuracy',
                                            'recall_micro',
                                            'precision_micro',
                                            'f1_micro'])

    log_this.info("Running Decision Tree cross validation on {} dataset".format(dataset_name))
    dt_cv_results = cross_validate(dt, X, y, cv=5, n_jobs=4,
                                   scoring=['accuracy',
                                            'recall_micro',
                                            'precision_micro',
                                            'f1_micro'])

    log_this.info("Running Random Forest cross validation on {} dataset".format(dataset_name))
    rf_cv_results = cross_validate(rf, X, y, cv=5, n_jobs=4,
                                   scoring=['accuracy',
                                            'recall_micro',
                                            'precision_micro',
                                            'f1_micro'])

    log_this.info("Running Support Vector Machine cross validation on {} dataset".format(dataset_name))
    svm_cv_results = cross_validate(svm, X, y, cv=5, n_jobs=4,
                                    scoring=['accuracy',
                                             'recall_micro',
                                             'precision_micro',
                                             'f1_micro'])

    log_this.info("Running Multi-layer Perceptron cross validation on {} dataset".format(dataset_name))
    mlp_cv_results = cross_validate(mlp, X, y, cv=5, n_jobs=4,
                                    scoring=['accuracy',
                                             'recall_micro',
                                             'precision_micro',
                                             'f1_micro'])

    all_results = [nb_cv_results, dt_cv_results, rf_cv_results, svm_cv_results, mlp_cv_results]
    return all_results
