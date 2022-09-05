import pandas as pd
import numpy as np
import os
import itertools


def get_all_csvs_from_folder(given_folder):
    num_files = len(os.listdir(given_folder))
    print("Folder contains {} files, beginning extraction...".format(num_files))
    data_frames = []
    for file in os.listdir(given_folder):
        location = given_folder + "/" + file
        data_frames.append(pd.read_csv(location))
    print("All files added to list of data frames...")
    return data_frames


def create_channel_sorted_csvs(data_frames):
    channels = ["phone_acc", "phone_gyr", "watch_acc", "watch_gyr"]
    all_combinations = []
    for i in range(1, len(channels)):
        for combination in itertools.combinations(channels, i):
            all_combinations.append(combination)
    print("All possible unique combinations of channels are as follows:\n {}".format(all_combinations))

    which_folder = 0
    print("beginning loop through data frames")
    for df in data_frames:
        folder_list = os.listdir("data/channel_analysis_datasets")
        print("working on folder {}".format(folder_list[which_folder]))
        for channel_set in all_combinations:
            save_location = "data/channel_analysis_datasets/" + folder_list[which_folder] + "/" + \
                            folder_list[which_folder][3:] + "_"
            column_list = []
            for channel in channel_set:
                save_location += channel + "_"
                include_columns = [col for col in df.columns if channel in col]
                column_list.extend(include_columns)
            column_list.extend(["target"])
            reduced_df = df[column_list]
            save_location += ".csv"
            reduced_df.to_csv(save_location, index=False)
            print("file saved in location: {}".format(save_location))
        which_folder += 1
