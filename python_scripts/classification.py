import random
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from logger import *
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
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


def run_optimised_classifiers(dataset, dataset_name):
    X, y = split_to_data_and_target(dataset)
    class_labels = np.unique(y)
    nb = naive_bayes()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    svm = SVC()
    mlp = MLPClassifier(hidden_layer_sizes=(int((108+len(class_labels)/2)), ), max_iter=1000)

    log_this.info("Computing cross validated metrics and confusion matrix for Naive Bayes on {} dataset".format(dataset_name))
    y_pred = cross_val_predict(nb, X, y, cv=10, n_jobs=4)
    nb_accuracy = accuracy_score(y, y_pred)
    nb_recall = recall_score(y, y_pred, average=None)
    nb_precision = precision_score(y, y_pred, average=None)
    nb_f1 = f1_score(y, y_pred, average=None)
    nb_cm = confusion_matrix(y, y_pred, labels=class_labels)

    log_this.info("Computing cross validated metrics and confusion matrix for Decision Tree on {} dataset".format(dataset_name))
    y_pred = cross_val_predict(dt, X, y, cv=10, n_jobs=4)
    dt_accuracy = accuracy_score(y, y_pred)
    dt_recall = recall_score(y, y_pred, average=None)
    dt_precision = precision_score(y, y_pred, average=None)
    dt_f1 = f1_score(y, y_pred, average=None)
    dt_cm = confusion_matrix(y, y_pred, labels=class_labels)

    log_this.info("Computing cross validated metrics and confusion matrix for Random Forest on {} dataset".format(dataset_name))
    y_pred = cross_val_predict(rf, X, y, cv=10, n_jobs=4)
    rf_accuracy = accuracy_score(y, y_pred)
    rf_recall = recall_score(y, y_pred, average=None)
    rf_precision = precision_score(y, y_pred, average=None)
    rf_f1 = f1_score(y, y_pred, average=None)
    rf_cm = confusion_matrix(y, y_pred, labels=class_labels)


    log_this.info("Computing cross validated metrics and confusion matrix for Support Vector Machine on {} dataset".format(dataset_name))
    y_pred = cross_val_predict(svm, X, y, cv=10, n_jobs=4)
    svm_accuracy = accuracy_score(y, y_pred)
    svm_recall = recall_score(y, y_pred, average=None)
    svm_precision = precision_score(y, y_pred, average=None)
    svm_f1 = f1_score(y, y_pred, average=None)
    svm_cm = confusion_matrix(y, y_pred, labels=class_labels)

    log_this.info("Computing cross validated metrics and confusion matrix for Multi-layer Perceptron on {} dataset".format(dataset_name))
    y_pred = cross_val_predict(mlp, X, y, cv=10, n_jobs=4)
    mlp_accuracy = accuracy_score(y, y_pred)
    mlp_recall = recall_score(y, y_pred, average=None)
    mlp_precision = precision_score(y, y_pred, average=None)
    mlp_f1 = f1_score(y, y_pred, average=None)
    mlp_cm = confusion_matrix(y, y_pred, labels=class_labels)

    all_results = {"naive_bayes": {"metrics": {"accuracy": nb_accuracy, "precision": nb_precision, "recall": nb_recall, "f1_score": nb_f1}, "confusion matrix": nb_cm},
                   "decision_tree": {"metrics":  {"accuracy": dt_accuracy, "precision": dt_precision, "recall": dt_recall, "f1_score": dt_f1}, "confusion matrix": dt_cm},
                   "random_forest": {"metrics":  {"accuracy": rf_accuracy, "precision": rf_precision, "recall": rf_recall, "f1_score": rf_f1}, "confusion matrix": rf_cm},
                   "support_vector_machine": {"metrics":  {"accuracy": svm_accuracy, "precision": svm_precision, "recall": svm_recall, "f1_score": svm_f1}, "confusion matrix": svm_cm},
                   "multi_layer_perceptron": {"metrics":  {"accuracy": mlp_accuracy, "precision": mlp_precision, "recall": mlp_recall, "f1_score": mlp_f1}, "confusion matrix": mlp_cm},
                   "class_labels": class_labels}
    return all_results
