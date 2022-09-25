import os
import pickle
import pandas as pd
from dataset_manipulation import *
from classification import *
from optimisation import *
from time import time
from logger import *
import tkinter as tk
from tkinter.messagebox import showinfo

log_this = custom_logger('log man')
os.chdir("C:/Users/cirid/OneDrive/Desktop/MPhil code repo/Inertial_motion_classification")


def run_fun():
    dfs = get_all_csvs_from_folder("data/complete_datasets")

    # create_channel_sorted_csvs(dfs)

    # flat_run(dfs)

    # run_parameter_optimisation(dfs, model_no=2)

    gs_results = load_optimisation_results("results/grid_search/")

    optimal_parameters = create_optimal_param_dict()

    all_results = []
    df_names = ["activity_recognition",
                "gender_recognition",
                "participant_recognition",
                "mixed_solo_activity_recognition",
                "pocket_solo_activity_recognition",
                "strap_solo_activity_recognition"]
    count = 0
    for data_frame in dfs:
        this_dataset_results = run_optimised_classifiers(data_frame, df_names[count], optimal_parameters)
        all_results.append(this_dataset_results)
        count += 1
    optimised_results = {'naive_bayes': all_results[0], 'decision_tree': all_results[1],
                         'random_forest': all_results[2], 'support_vector_machine': all_results[3],
                         'multi_layer_perceptron': all_results[4]}

    # save_cv_results(gs_results)

    log_this.info("finished")



def pop_up_please(window_title, message):
    showinfo(window_title, message)


def flat_run(data_frames):
    flat_run_all_classifiers(data_frames[0], "group activity recognition")
    flat_run_all_classifiers(data_frames[1], "group gender recognition")
    flat_run_all_classifiers(data_frames[2], "group participant recognition")
    flat_run_all_classifiers(data_frames[3], "solo mixed location activity recognition")
    flat_run_all_classifiers(data_frames[4], "solo pocket location activity recognition")
    flat_run_all_classifiers(data_frames[5], "solo strap location activity recognition")


def run_parameter_optimisation(dfs, model_no=0, all=False):
    df_names = ['activity_recognition',
                'gender_recognition',
                'participant_recognition',
                'mixed_solo_activity_recognition',
                'pocket_solo_activity_recognition',
                'strap_solo_activity_recognition']

    model_names = ['decision tree',
                   'random forest',
                   'support vector machine',
                   'multi-layer perceptron']

    count = 0
    tock = time()
    for data_frame in dfs:
        if model_no == 0 or all:
            log_this.info("Running grid search for {} classifier, on {} dataset".format(model_names[0], df_names[count]))
            dt, scorer_list, parameter_dict = set_up_for_decision_tree()
            dt_grid_search_results = run_grid_search(data_frame, dt, parameter_dict, scorer_list)
            save_location = "results/grid_search/decision_tree/decision_tree_" + df_names[count] + ".pickle"
            log_this.info("saving result")
            with open(save_location, 'wb') as f:
                pickle.dump(dt_grid_search_results, f)
                f.close()
            log_this.info("result saved")
            # pop_up_please("{} grid search".format(model_names[0]), "finished run on {} dataset".format(df_names[count]))

        if model_no == 1 or all:
            log_this.info("Running grid search for {} classifier, on {} dataset".format(model_names[1], df_names[count]))
            rf, scorer_list, parameter_dict = set_up_for_random_forest()
            rf_grid_search_results = run_grid_search(data_frame, rf, parameter_dict, scorer_list)
            save_location = "results/grid_search/random_forest/random_forest_" + df_names[count] + ".pickle"
            log_this.info("saving result")
            with open(save_location, 'wb') as f:
                pickle.dump(rf_grid_search_results, f)
                f.close()
            log_this.info("result saved")
            # pop_up_please("{} grid search".format(model_names[1]), "finished run on {} dataset".format(df_names[count]))

        if model_no == 2 or all:
            log_this.info("Running grid search for {} classifier, on {} dataset".format(model_names[2], df_names[count]))
            svm, scorer_list, parameter_dict = set_up_for_support_vector_machine()
            svm_grid_search_results = run_grid_search(data_frame, svm, parameter_dict, scorer_list)
            save_location = "results/grid_search/support_vector_machine/support_vector_machine_" + df_names[count] + ".pickle"
            log_this.info("saving result")
            with open(save_location, 'wb') as f:
                pickle.dump(svm_grid_search_results, f)
                f.close()
            log_this.info("result saved")
            # pop_up_please("{} grid search".format(model_names[2]), "finished run on {} dataset".format(df_names[count]))

        if model_no == 3 or all:
            log_this.info("Running grid search for {} classifier, on {} dataset".format(model_names[3], df_names[count]))
            mlp, scorer_list, parameter_dict = set_up_for_multi_layer_perceptron()
            mlp_grid_search_results = run_grid_search(data_frame, mlp, parameter_dict, scorer_list)
            save_location = "results/grid_search/multi_layer_perceptron/multi_layer_perceptron_" + df_names[count] + ".pickle"
            log_this.info("saving result")
            with open(save_location, 'wb') as f:
                pickle.dump(mlp_grid_search_results, f)
                f.close()
            log_this.info("result saved")
            # pop_up_please("{} grid search".format(model_names[3]), "finished run on {} dataset".format(df_names[count]))

        count += 1

    tick = time()
    elapsed_time = tick - tock
    elapsed_seconds = int(elapsed_time % 60)
    elapsed_minutes = int((elapsed_time % 3600) / 60)
    elapsed_hours = int(elapsed_time / 3600)
    log_this.info("The total elapsed time for running all datasets was {} hours {} minutes and {} seconds"
                  .format(elapsed_hours, elapsed_minutes, elapsed_seconds))


def load_optimisation_results(path_to_results):
    results = []
    folders = os.listdir(path_to_results)
    for folder in folders:
        current_path = path_to_results + folder + "/"
        files = os.listdir(current_path)
        for file in files:
            current_path += file
            current_result = load_pickle_files(current_path)
            results.append(current_result)
            current_path = path_to_results + folder + "/"
    return results


def save_cv_results(grid_search_results):
    df_names = ['activity_recognition',
                'gender_recognition',
                'participant_recognition',
                'mixed_solo_activity_recognition',
                'pocket_solo_activity_recognition',
                'strap_solo_activity_recognition']

    model_names = ['decision_tree',
                   'multi_layer_perceptron',
                   'random_forest',
                   'support_vector_machine']

    count_model = 0
    count_dataset = 0
    for result_set in grid_search_results:
        path = "results/grid_search_cv_results/{}/{}_{}_cv_results.csv".format(model_names[count_model],
                                                                               model_names[count_model],
                                                                               df_names[count_dataset])
        cv_results = result_set.cv_results_
        panda_results = pd.DataFrame.from_dict(cv_results)
        panda_results.to_csv(path, index=False)
        count_dataset += 1
        if count_dataset > 5:
            count_model += 1
            count_dataset = 0


def create_optimal_param_dict():
    df_names = ['activity_recognition',
                'gender_recognition',
                'participant_recognition',
                'mixed_solo_activity_recognition',
                'pocket_solo_activity_recognition',
                'strap_solo_activity_recognition']

    model_names = ['decision_tree',
                   'multi_layer_perceptron',
                   'random_forest',
                   'support_vector_machine']
    temp_dict = {}

    for model in model_names:
        for dataset in df_names:
            key_name = "{}_{}".format(model, dataset)
            temp_dict[key_name] = {}
            if model == "decision_tree":
                temp_dict[key_name]["ccp_alpha"] = 0.0
                temp_dict[key_name]["criterion"] = "gini"
                temp_dict[key_name]["min_impurity_decrease"] = 0.0
                temp_dict[key_name]["splitter"] = "best"
            if model == "multi_layer_perceptron":
                temp_dict[key_name]["alpha"] = 0.001
                temp_dict[key_name]["hidden_layer_sizes"] = (100,)
                temp_dict[key_name]["learning_rate"] = "constant"
                temp_dict[key_name]["learning_rate_init"] = 0.0005
                temp_dict[key_name]["max_iter"] = 1000
            if model == "random_forest":
                temp_dict[key_name]["ccp_alpha"] = 0.0
                temp_dict[key_name]["criterion"] = "gini"
                temp_dict[key_name]["min_impurity_decrease"] = 0.0
                temp_dict[key_name]["n_estimators"] = 500
            if model == "support_vector_machine":
                temp_dict[key_name]["C"] = 1.0
                temp_dict[key_name]["decision_function_shape"] = "ovo"
                temp_dict[key_name]["degree"] = 4
                temp_dict[key_name]["gamma"] = "scale"
                temp_dict[key_name]["kernel"] = "poly"
                temp_dict[key_name]["probability"] = True
                temp_dict[key_name]["tol"] = 0.001

    file_location = "results/grid_search_cv_results/"
    dt_locs = os.listdir(file_location + model_names[0])
    mlp_locs = os.listdir(file_location + model_names[1])
    rf_locs = os.listdir(file_location + model_names[2])
    svm_locs = os.listdir(file_location + model_names[3])
    all_locs = [dt_locs, mlp_locs, rf_locs, svm_locs]

    df_list = []

    for loc_list in all_locs:
        for file in loc_list:
            for model in model_names:
                if model in file:
                    location = file_location + model + "/" + file
                    current_df = pd.read_csv(location)
                    sorted_df = current_df.sort_values("rank_test_accuracy", ignore_index=True)
                    for dataset in df_names:
                        if dataset in file:
                            key_name = "{}_{}".format(model, dataset)
                            if model == "decision_tree":
                                temp_dict[key_name]["ccp_alpha"] = sorted_df["param_ccp_alpha"].iloc[0]
                                temp_dict[key_name]["criterion"] = sorted_df["param_criterion"].iloc[0]
                                temp_dict[key_name]["min_impurity_decrease"] = sorted_df["param_min_impurity_decrease"].iloc[0]
                                temp_dict[key_name]["splitter"] = sorted_df["param_splitter"].iloc[0]
                            if model == "multi_layer_perceptron":
                                temp_dict[key_name]["alpha"] = sorted_df["param_alpha"].iloc[0]
                                temp_dict[key_name]["hidden_layer_sizes"] = sorted_df["param_hidden_layer_sizes"].iloc[0]
                                temp_dict[key_name]["learning_rate"] = sorted_df["param_learning_rate"].iloc[0]
                                temp_dict[key_name]["learning_rate_init"] = sorted_df["param_learning_rate_init"].iloc[0]
                                temp_dict[key_name]["max_iter"] = sorted_df["param_max_iter"].iloc[0]
                            if model == "random_forest":
                                temp_dict[key_name]["ccp_alpha"] = sorted_df["param_ccp_alpha"].iloc[0]
                                temp_dict[key_name]["criterion"] = sorted_df["param_criterion"].iloc[0]
                                temp_dict[key_name]["min_impurity_decrease"] = sorted_df["param_min_impurity_decrease"].iloc[0]
                                temp_dict[key_name]["n_estimators"] = sorted_df["param_n_estimators"].iloc[0]
                            if model == "support_vector_machine":
                                temp_dict[key_name]["C"] = sorted_df["param_C"].iloc[0]
                                temp_dict[key_name]["decision_function_shape"] = sorted_df["param_decision_function_shape"].iloc[0]
                                temp_dict[key_name]["degree"] = sorted_df["param_degree"].iloc[0]
                                temp_dict[key_name]["gamma"] = sorted_df["param_gamma"].iloc[0]
                                temp_dict[key_name]["kernel"] = sorted_df["param_kernel"].iloc[0]
                                temp_dict[key_name]["probability"] = sorted_df["param_probability"].iloc[0]
                                temp_dict[key_name]["tol"] = sorted_df["param_tol"].iloc[0]
                            continue
    return temp_dict


if __name__ == '__main__':
    run_fun()
