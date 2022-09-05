import os
from dataset_manipulation import *
from classification import *


def run_fun():
    dfs = get_all_csvs_from_folder("data/complete_datasets")
    create_channel_sorted_csvs(dfs)
    flat_run(dfs)


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
