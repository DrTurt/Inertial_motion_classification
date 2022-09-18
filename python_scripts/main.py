import os
import pickle
from dataset_manipulation import *
from classification import *
from optimisation import *
from time import time
from logger import *

log_this = custom_logger('log man')
os.chdir("C:/Users/cirid/OneDrive/Desktop/MPhil code repo/Inertial_motion_classification")


def run_fun():
    dfs = get_all_csvs_from_folder("data/complete_datasets")

    # create_channel_sorted_csvs(dfs)

    # flat_run(dfs)

    run_parameter_optimisation(dfs, model_no=3)

    gs_results = load_optimisation_results("results/")


def flat_run(data_frames):
    flat_run_all_classifiers(data_frames[0], "group activity recognition")
    flat_run_all_classifiers(data_frames[1], "group gender recognition")
    flat_run_all_classifiers(data_frames[2], "group participant recognition")
    flat_run_all_classifiers(data_frames[3], "solo mixed location activity recognition")
    flat_run_all_classifiers(data_frames[4], "solo pocket location activity recognition")
    flat_run_all_classifiers(data_frames[5], "solo strap location activity recognition")


def run_parameter_optimisation(dfs, model_no=0, all=False):
    dfs = dfs[1:]
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

    count = 1
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
        for file in folder:
            current_path += file
            current_result = load_pickle_files(current_path)
            results.append(current_result)
            current_path = path_to_results + folder + "/"
    return results


if __name__ == '__main__':
    run_fun()
