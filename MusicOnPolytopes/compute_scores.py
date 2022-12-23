# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:21:07 2021

@author: amarmore
"""

from IPython.display import display, Markdown

# Self-code imports
import polytopes.segmentation_algorithms as algos
import polytopes.data_manipulation as dm


#Generic imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

database_path = "C:/Users/amarmore/Desktop/Projects/RWC_annotations/final_bimbot_al/"
annotations_folder_path = "C:/Users/amarmore/Desktop/Audio samples/RWC Pop/annotations/final_bimbot/"

def compute_scores_for_different_penalties(database, database_name, segments_size_range, polytope_irr_range, relation_type, max_size = 42):
    """
    Compute the scores for different penalty values and print the in readable format.

    Parameters
    ----------
    database : TYPE
        DESCRIPTION.
    segments_size_range : TYPE
        DESCRIPTION.
    polytope_irr_range : TYPE
        DESCRIPTION.
    relation_type : TYPE
        DESCRIPTION.
    max_size : TYPE, optional
        DESCRIPTION. The default is 42.

    Returns
    -------
    None.

    """

    all_scores = math.inf * np.ones((len(database), len(segments_size_range), len(polytope_irr_range), 2, 3))
    
    for idx_song, song in enumerate(database):
        #time_start = time.time()
        song_number = song.split(".")[0]
        #print(song_number)
        bag_of_chords = dm.flowify_song(database_path + song)
    
        annot_name = "{:03d}.manual.seg".format(int(song_number))
        annotation_file = open(annotations_folder_path + annot_name,'r')
        annotation = annotation_file.read().replace("\n", "").split(" ")
        annotation = np.array([int(x) - 1 for x in annotation])
        beat_indexed_annotation = np.array(dm.frontiers_to_segments(annotation))
        
        for idx_seg, segment_size_penalty in enumerate(segments_size_range):
            for idx_irr, polytope_irregularity_penalty in enumerate(polytope_irr_range):
    
                frontiers, cost = algos.dynamic_minimization_guichaoua_persist_segments(bag_of_chords, database_name,
                                                                                segment_size_penalty = segment_size_penalty,
                                                                                min_size = 8, max_size = max_size, target_size = 32, 
                                                                                polytope_irregularity_penalty = polytope_irregularity_penalty,
                                                                                relation_type = relation_type, song_number = song_number)
    
                #for fst, snd in zip(frontiers[:-1], frontiers[1:]):
                #    g_distrib_segments.append(snd - fst)
    
                #Scores, computed on the beat annotation
                beat_indexed_segments = dm.frontiers_to_segments(frontiers)
    
                all_scores[idx_song, idx_seg, idx_irr, 0] = dm.compute_score_of_segmentation(beat_indexed_annotation, beat_indexed_segments, 
                                                                           window_length = 0.2)
                all_scores[idx_song, idx_seg, idx_irr, 1] = dm.compute_score_of_segmentation(beat_indexed_annotation, beat_indexed_segments, 
                                                                           window_length = 3)
        
    """print_res_for_songs_range(g_all_res_zero_tol, context = "Zero Tolerance")
    print_res_for_songs_range(g_all_res_three_tol, context = "Three Tolerance")
    plt.figure(figsize=(15,5))
    plt.hist(g_distrib_segments, bins = range(max_size))
    plt.xlabel("Segment's size (in nb of beats)")
    #plt.title("Prec: {}, rap: {}, F measure: {}".format(prec, rap, fmes))
    plt.plot()"""
    
    lines = np.array(['Precision', 'Recall', 'F measure'])  
    
    all_scores_np = np.array(all_scores)
    arr_zero_five = []
    arr_three = []
    
    col_seg = []
    col_irr = []
    for i_seg in range(len(segments_size_range)):
        for i_irr in range(len(polytope_irr_range)):
            line_zero_five = []
            line_three = []
            for j in range(3):
                line_zero_five.append(round(np.mean(all_scores_np[:,i_seg, i_irr, 0, j]),4))
                line_three.append(round(np.mean(all_scores_np[:,i_seg, i_irr, 1, j]),4))
            arr_zero_five.append(line_zero_five)
            arr_three.append(line_three)
            col_seg.append(f"Size penalty: {segments_size_range[i_seg]}")
            col_irr.append(f"Polytope penalty: {polytope_irr_range[i_irr]}")
    
    col = [np.array(col_seg),np.array(col_irr)]
    
    dataframe = pd.DataFrame(arr_zero_five, index=col, columns=lines)
    display(dataframe.style.bar(subset=["F measure"], color='#5fba7d'))
    
    dataframe = pd.DataFrame(arr_three, index=col, columns=lines)
    display(dataframe.style.bar(subset=["F measure"], color='#5fba7d'))