import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    precision_recall_curve, roc_auc_score, f1_score


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
    print("Naive Bayes classifier accuracy on {} dataset: {}".format(dataset_name, score))

    dt.fit(X_train, y_train)
    score = dt.score(X_test, y_test)
    print("Decision tree classifier accuracy on {} dataset: {}".format(dataset_name, score))

    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    print("Random forest classifier accuracy on {} dataset: {}".format(dataset_name, score))

    svm.fit(X_train, y_train)
    score = svm.score(X_test, y_test)
    print("Support vector machine classifier accuracy on {} dataset: {}".format(dataset_name, score))

    mlp.fit(X_train, y_train)
    score = mlp.score(X_test, y_test)
    print("Multi-layer perceptron classifier accuracy on {} dataset: {}".format(dataset_name, score))
