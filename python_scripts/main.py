import os
import pickle
from dataset_manipulation import *
from classification import *
from optimisation import *


def run_fun():
    dfs = get_all_csvs_from_folder("data/complete_datasets")
    # create_channel_sorted_csvs(dfs)
    # flat_run(dfs)
    df_names = ['activity_recognition',
                'gender_recognition',
                'participant_recognition',
                'mixed_solo_activity_recognition',
                'pocket_solo_activity_recognition',
                'strap_solo_activity_recognition']
    count = 0
    for data_frame in dfs:
        dt, scorer_list, parameter_dict = set_up_for_decision_tree()
        dt_grid_search_results = run_grid_search(data_frame, dt, parameter_dict, scorer_list)
        save_location = "results/grid_search/decision_tree/decision_tree_" + df_names[count] + ".pickle"
        with open(save_location, 'wb') as f:
            pickle.dump(dt_grid_search_results, f)
            f.close()

        rf, scorer_list, parameter_dict = set_up_for_random_forest()
        rf_grid_search_results = run_grid_search(data_frame, rf, parameter_dict, scorer_list)
        save_location = "results/grid_search/random_forest/random_forest_" + df_names[count] + ".pickle"
        with open(save_location, 'wb') as f:
            pickle.dump(rf_grid_search_results, f)
            f.close()

        svm, scorer_list, parameter_dict = set_up_for_support_vector_machine()
        svm_grid_search_results = run_grid_search(data_frame, svm, parameter_dict, scorer_list)
        save_location = "results/grid_search/support_vector_machine/support_vector_machine_" + df_names[count] + ".pickle"
        with open(save_location, 'wb') as f:
            pickle.dump(svm_grid_search_results, f)
            f.close()

        mlp, scorer_list, parameter_dict = set_up_for_multi_layer_perceptron()
        mlp_grid_search_results = run_grid_search(data_frame, mlp, parameter_dict, scorer_list)
        save_location = "results/grid_search/multi_later_perceptron/multi_layer_perceptron_" + df_names[count] + ".pickle"
        with open(save_location, 'wb') as f:
            pickle.dump(mlp_grid_search_results, f)
            f.close()

        count += 1

    print("break")


def flat_run(data_frames):
    flat_run_all_classifiers(data_frames[0], "group activity recognition")
    flat_run_all_classifiers(data_frames[1], "group gender recognition")
    flat_run_all_classifiers(data_frames[2], "group participant recognition")
    flat_run_all_classifiers(data_frames[3], "solo mixed location activity recognition")
    flat_run_all_classifiers(data_frames[4], "solo pocket location activity recognition")
    flat_run_all_classifiers(data_frames[5], "solo strap location activity recognition")


if __name__ == '__main__':
    os.chdir("C:/Users/cirid/OneDrive/Desktop/MPhil code repo/Inertial_motion_classification")
    run_fun()
