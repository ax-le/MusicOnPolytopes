# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:08:18 2019

@author: amarmore
"""

import os

# Self-code imports
import polytopes.baselines as baselines
import polytopes.data_manipulation as dm

#Generic imports
import numpy as np

database_path = "C:\\Users\\amarmore\\Desktop\\Projects\\RWC_annotations\\final_bimbot_al\\"
annotations_folder_path = "C:\\Users\\amarmore\\Desktop\\Audio samples\\RWC Pop\\annotations\\final_bimbot\\"
persisted_path = "C:\\Users\\amarmore\\Desktop\\data_persisted"
notebook_dir = "C:\\Users\\amarmore\\Desktop\\Projects\\PhD main projects\\On Git\\Code\\Polytopes and relation\\Notebooks"

if __name__ == '__main__':
    max_size = 42
    manual = []
    for file in os.listdir(database_path):
        bag_of_words = file.split(".")
        if bag_of_words[-1] == "seq":
            if bag_of_words[-3] == "manual":
                manual.append(file)
    
    for song in manual:
        print("Current song: {}".format(song))
        song_number = song.split(".")[0]
        bag_of_chords = dm.flowify_song(database_path + song)

        try:
            frontiers = np.load("{}\\persisted_content\\estimated_frontiers\\guichaoua_song{}_penalties{}_maxsize{}.npy".format(notebook_dir, song_number, (0,0), max_size), allow_pickle = True)
        except FileNotFoundError:
            frontiers, cost = baselines.dynamic_minimization_guichaoua_phdlike(bag_of_chords, positive_segment_size_penalty = 0, negative_segment_size_penalty = 0, min_size = 8, max_size = max_size, target_size = 32, positive_penalty = 0, negative_penalty = 0)
            fro = np.array(frontiers, dtype=object)
            np.save("{}\\persisted_content\\estimated_frontiers\\guichaoua_song{}_penalties{}_maxsize{}".format(notebook_dir, song_number, (0,0), max_size), fro)
