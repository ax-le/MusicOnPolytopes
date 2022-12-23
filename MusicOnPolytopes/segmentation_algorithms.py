# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:14:22 2021

@author: amarmore
"""
import math
import time
import warnings

from polytopes.model.chord import Chord
import polytopes.data_manipulation as dm
import polytopes.polytopical_costs as pc
import polytopes.pattern_factory as pf
import polytopes.model.errors as err
import polytopes.segmentation_helper as sh
import numpy as np
import os

"""
File for all segmentation algorithms, i.e. different dynamic programming algorithms, associated with different codes.
TODO: when the code is stable, I should keep only one dynamic programming algorithm and specify the cost with an argument.
"""

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# %% Costs with implication system
## Guichaoua original cost
def dynamic_minimization_guichaoua(chord_sequence, positive_penalty = 2.25, negative_penalty = 3, min_size = 8, max_size = 40, positive_segment_size_penalty = 0, negative_segment_size_penalty = 0.125, target_size = 32, global_antecedents = False, persist_patterns = True, relation_type = "triad_circle", song_for_persist = None):
    """
    Dynamic programming algorithm applied to the search of the optimal segmentation of a piece of music, with Guichaoua's cost.
    
    Given the costs of all possible segments in the piece (flow), it finds the segmentation which minimizes the gloabl sum of all segment costs.
    TODO: add a reference to explain this cost.

    Parameters
    ----------
    chord_sequence : list of chords, of any type
        The chord sequence (the song) to segment.
    positive_penalty, negative_penalty : float/integer, optional
        Penalty parameter related to irregularities of the polytope.
        Positive corresponds to the penalty when the polytope contains addition, negative is for deletion.
        They are constants and not function of the size of irregularities.
    min_size : integer, optional
        Minimal size for a segment (except the first one and the last one). The default is 8.
    max_size : integer, optional
        Maximal size for a segment. The default is 40.
    positive_segment_size_penalty, negative_segment_size_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the size of the segment.
        positive_segment_size_penalty is the parameter when size exceeds 'target_size', negative is for size shorter than 'target_size'.
    target_size : integer, optional
        The optimal size, used for the penalty related to the segment size. 
        The default is 32.
    global_antecedents : boolean, optional
        A boolean to indicate whether antecedents should be the directs ones (i.e. the ones which are linked with an arrow to the actual element),
        or global ones (i.e. antecedents where a path exists with the current element, like in Guichaoua's exact paradigm).
        In my understanding of the model, direct antecedents should be considered as the default behavior.
    persist_patterns : boolean, optional
        A boolean, used to decide whether patterns should be computed once, and then reused (True) or not (False).
        If True, the patterns (and other informations such as antecedents of every element for instance) are persisted on a file on the machine running the code at first computation.
        (NB: I tried to store them as a variable in a huge list at first, but it resulted in errors, probably due to the size of all patterns...)
        If False, they are computed at each iteration (so for each element in each possible size).
        I strongly encourage to set it to True, as it reduces the computation time by a factor of 10.
        The default is True.

    Raises
    ------
    err
        Errors to avoid bugs at runtime (ToDebugException) or invalid arguments (InvalidArgumentValueException).

    Returns
    -------
    frontiers : list of integers
        The estimated frontiers for this segmentation.
    cost : integer
        The total cost of this segmentation.

    """
    if min_size < 2:
        raise err.InvalidArgumentValueException("Minimum size should be at least 2.")
    costs = [math.inf for i in range(len(chord_sequence))]
    costs[0] = 0
    
    symbolic_flow = []
    for chord in chord_sequence:
        if dm.is_a_chord_object(chord):
            symbolic_flow.append(chord.triad)
        else:
            symbolic_flow.append(chord)
            #symbolic_flow.append(Chord(chord).triad)

    segments_best_starts = [None for i in range(len(chord_sequence))]
    segments_best_starts[0] = 0   
    
    for current_idx in range(2, len(symbolic_flow)):
        if current_idx < min_size:
            possible_starts = [0]
        elif current_idx == len(symbolic_flow) - 1:
            possible_starts = sh.possible_segment_start(current_idx, min_size = 2, max_size = max_size)
        else:
            possible_starts = sh.possible_segment_start(current_idx, min_size = min_size, max_size = max_size)
        for possible_start_idx in possible_starts:
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")
            segment = [symbolic_flow[k] for k in range(possible_start_idx, current_idx + 1)]
            segment_size = len(segment)
            
            if persist_patterns:
                if global_antecedents:
                    try:
                        this_bag = np.load("{}/persisted_content/compute_patterns_with_global_antecedents_for_size_{}.npy".format(CURR_DIR, segment_size), allow_pickle = True)
                    except FileNotFoundError:
                        this_bag = sh.compute_patterns_with_global_antecedents_for_size(segment_size)
                        arr = np.array(this_bag, dtype=object)
                        np.save("{}/persisted_content/compute_patterns_with_global_antecedents_for_size_{}".format(CURR_DIR, segment_size), arr)            
                else:
                    try:
                        this_bag = np.load("{}/persisted_content/compute_patterns_with_antecedents_for_size_{}.npy".format(CURR_DIR, segment_size), allow_pickle = True)
                    except FileNotFoundError:
                        this_bag = sh.compute_patterns_with_antecedents_for_size(segment_size)
                        arr = np.array(this_bag, dtype=object)
                        np.save("{}/persisted_content/compute_patterns_with_antecedents_for_size_{}".format(CURR_DIR, segment_size), arr)
            else:
                if global_antecedents:
                    this_bag = sh.compute_patterns_with_global_antecedents_for_size(segment_size)
                else:
                    this_bag = sh.compute_patterns_with_antecedents_for_size(segment_size)

            if this_bag != []:                
                this_segment_cost = math.inf
                for a_pattern in this_bag:
                    if global_antecedents:
                        this_polytope_cost = pc.guichaoua_cost_global_antecedents_successors(segment, a_pattern[0], a_pattern[3], a_pattern[4], a_pattern[5], current_min = this_segment_cost, relation_type = relation_type)
                    else:
                        this_polytope_cost = pc.guichaoua_cost(segment, a_pattern[0], a_pattern[3], a_pattern[4], a_pattern[5], current_min = this_segment_cost, relation_type = relation_type)

                    this_polytope_cost += pc.irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = positive_penalty, negative_penalty = negative_penalty)
                    if this_polytope_cost < this_segment_cost:
                        this_segment_cost = this_polytope_cost

            else:
                warnings.warn("No Polytope is available for this size of segment. Trying sequential cost instead.")
                this_segment_cost = pc.sequential_score(segment, 0, relation_type = relation_type)
                #this_segment_cost = math.inf

            this_segment_cost += sh.penalty_cost_guichaoua(segment_size, target_size = target_size, positive_segment_size_penalty = positive_segment_size_penalty, negative_segment_size_penalty = negative_segment_size_penalty)
                
            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost < costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0

            else:
                if costs[possible_start_idx - 1] + this_segment_cost < costs[current_idx]:
                    # Optimal cost until previous segment + cost of this segment.
                    costs[current_idx] = costs[possible_start_idx - 1] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx

    frontiers = [len(chord_sequence)] #  Because frontiers are start of next segment, so it should be the chord after the next one.
    best_start_for_this_segment = segments_best_starts[len(chord_sequence) - 1]
    while best_start_for_this_segment > 0: # If best_start_for_this_segment == None, an error is raised.
        frontiers.append(best_start_for_this_segment)
        precedent_end = best_start_for_this_segment - 1 # Because previous segment ends at the chord before this one.
        best_start_for_this_segment = segments_best_starts[precedent_end]
        if precedent_end == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    frontiers.append(0) # Frontiers are here the start of a new segment, the first chord of a segment.
    return frontiers[::-1], costs[-1]

# Persisting computations, to gain time
def dynamic_minimization_guichaoua_persist_segments(chord_sequence, database, polytope_irregularity_penalty = 2, polytope_irregularity_function = "guichaoua", min_size = 8, max_size = 40, segment_size_penalty = 1, target_size = 32, relation_type = "triad_circle", song_number = None):
    """
    Dynamic programming algorithm applied to the search of the optimal segmentation of a piece of music, with Guichaoua's cost.
    
    Given the costs of all possible segments in the piece (flow), it finds the segmentation which minimizes the gloabl sum of all segment costs.
    TODO: add a reference to explain this cost.

    Parameters
    ----------
    chord_sequence : list of chords, of any type
        The chord sequence (the song) to segment.
    positive_penalty, negative_penalty : float/integer, optional
        Penalty parameter related to irregularities of the polytope.
        Positive corresponds to the penalty when the polytope contains addition, negative is for deletion.
        They are constants and not function of the size of irregularities.
    min_size : integer, optional
        Minimal size for a segment (except the first one and the last one). The default is 8.
    max_size : integer, optional
        Maximal size for a segment. The default is 40.
    positive_segment_size_penalty, negative_segment_size_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the size of the segment.
        positive_segment_size_penalty is the parameter when size exceeds 'target_size', negative is for size shorter than 'target_size'.
    target_size : integer, optional
        The optimal size, used for the penalty related to the segment size. 
        The default is 32.
    global_antecedents : boolean, optional
        A boolean to indicate whether antecedents should be the directs ones (i.e. the ones which are linked with an arrow to the actual element),
        or global ones (i.e. antecedents where a path exists with the current element, like in Guichaoua's exact paradigm).
        In my understanding of the model, direct antecedents should be considered as the default behavior.
    persist_patterns : boolean, optional
        A boolean, used to decide whether patterns should be computed once, and then reused (True) or not (False).
        If True, the patterns (and other informations such as antecedents of every element for instance) are persisted on a file on the machine running the code at first computation.
        (NB: I tried to store them as a variable in a huge list at first, but it resulted in errors, probably due to the size of all patterns...)
        If False, they are computed at each iteration (so for each element in each possible size).
        I strongly encourage to set it to True, as it reduces the computation time by a factor of 10.
        The default is True.

    Raises
    ------
    err
        Errors to avoid bugs at runtime (ToDebugException) or invalid arguments (InvalidArgumentValueException).

    Returns
    -------
    frontiers : list of integers
        The estimated frontiers for this segmentation.
    cost : integer
        The total cost of this segmentation.

    """
    if min_size < 2:
        raise err.InvalidArgumentValueException("Minimum size should be at least 2.")
    costs = [math.inf for i in range(len(chord_sequence))]
    costs[0] = 0
    
    symbolic_flow = []
    for chord in chord_sequence:
        if dm.is_a_chord_object(chord):
            symbolic_flow.append(chord.triad)
        else:
            symbolic_flow.append(chord)
            #symbolic_flow.append(Chord(chord).triad)

    segments_best_starts = [None for i in range(len(chord_sequence))]
    segments_best_starts[0] = 0
    
    if song_number == None:
        raise NotImplementedError("Can't precompute or load anything")
    
    try:
        this_song_optimal_costs = np.load("{}/persisted_content/guichaoua_song_costs_{}/guichaoua_song{}_relation{}_irreg_function{}_irreg_val{}.npy".format(CURR_DIR, database, song_number, relation_type, polytope_irregularity_function, polytope_irregularity_penalty), allow_pickle = True)
        costs_loaded = True
    except FileNotFoundError:
        costs_loaded = False
        to_save_optimal_costs = -1 * np.ones((len(chord_sequence),len(chord_sequence)))
    
    for current_idx in range(2, len(symbolic_flow)):
        if current_idx < min_size:
            possible_starts = [0]
        elif current_idx == len(symbolic_flow) - 1:
            possible_starts = sh.possible_segment_start(current_idx, min_size = 2, max_size = max_size)
        else:
            possible_starts = sh.possible_segment_start(current_idx, min_size = min_size, max_size = max_size)
        for possible_start_idx in possible_starts:
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")
            segment = [symbolic_flow[k] for k in range(possible_start_idx, current_idx + 1)]
            segment_size = len(segment)
            
            if costs_loaded:
                this_segment_cost = this_song_optimal_costs[possible_start_idx,current_idx]
                if this_segment_cost == -1 or this_segment_cost == math.inf:
                    raise err.ToDebugException("Error here, invalid value persisted")
            else:
                try:
                    this_bag = np.load("{}/persisted_content/compute_patterns_with_antecedents_for_size_{}.npy".format(CURR_DIR, segment_size), allow_pickle = True)
                except FileNotFoundError:
                    this_bag = sh.compute_patterns_with_antecedents_for_size(segment_size)
                    arr = np.array(this_bag, dtype=object)
                    np.save("{}/persisted_content/compute_patterns_with_antecedents_for_size_{}".format(CURR_DIR, segment_size), arr)
    
                if this_bag != []:
                    it_sgt = ''
                    for chord in segment:
                        it_sgt += chord.replace('b','-')
                    try:
                        all_costs = np.load("{}/persisted_content/guichaoua_pc_costs_{}/guichaoua_costs_seg{}_relation{}.npy".format(CURR_DIR, database, it_sgt, relation_type), allow_pickle = True)
                    except FileNotFoundError:
                        all_costs = [pc.guichaoua_cost(segment, a_pattern[0], a_pattern[3], a_pattern[4], a_pattern[5], current_min = math.inf, relation_type = relation_type) for a_pattern in this_bag]
                        #arr_costs = np.array(all_costs, dtype=object)
                        #np.save("{}\\persisted_content\\guichaoua_pc_costs\\guichaoua_costs_seg{}_relation{}".format(CURR_DIR, it_sgt, relation_type), arr_costs)
                    this_segment_cost = math.inf
                    for idx, a_pattern in enumerate(this_bag):
                        this_polytope_cost = all_costs[idx]
                        if polytope_irregularity_function == "guichaoua":
                            this_polytope_cost += pc.irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = polytope_irregularity_penalty, negative_penalty = polytope_irregularity_penalty)
                        else:
                            raise NotImplementedError("No other irregularity function is implemented")
                        if this_polytope_cost < this_segment_cost:
                            this_segment_cost = this_polytope_cost
    
                else:
                    warnings.warn("No Polytope is available for this size of segment. Trying sequential cost instead.")
                    this_segment_cost = pc.sequential_score(segment, 0, relation_type = relation_type)
                    #this_segment_cost = math.inf
                
                to_save_optimal_costs[possible_start_idx, current_idx] = this_segment_cost
                if this_segment_cost == math.inf or this_segment_cost == -1:
                    raise err.ToDebugException("Invalid segment cost to persist")

            this_segment_cost += sh.penalty_cost_guichaoua(segment_size, target_size = target_size, positive_segment_size_penalty = segment_size_penalty, negative_segment_size_penalty = segment_size_penalty)
                
            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost < costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0

            else:
                if costs[possible_start_idx - 1] + this_segment_cost < costs[current_idx]:
                    # Optimal cost until previous segment + cost of this segment.
                    costs[current_idx] = costs[possible_start_idx - 1] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx
                    
    if not costs_loaded:
        np.save("{}/persisted_content/guichaoua_song_costs_{}/guichaoua_song{}_relation{}_irreg_function{}_irreg_val{}".format(CURR_DIR, database, song_number, relation_type, polytope_irregularity_function, polytope_irregularity_penalty), to_save_optimal_costs)

    frontiers = [len(chord_sequence)] #  Because frontiers are start of next segment, so it should be the chord after the next one.
    best_start_for_this_segment = segments_best_starts[len(chord_sequence) - 1]
    while best_start_for_this_segment > 0: # If best_start_for_this_segment == None, an error is raised.
        frontiers.append(best_start_for_this_segment)
        precedent_end = best_start_for_this_segment - 1 # Because previous segment ends at the chord before this one.
        best_start_for_this_segment = segments_best_starts[precedent_end]
        if precedent_end == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    frontiers.append(0) # Frontiers are here the start of a new segment, the first chord of a segment.
    return frontiers[::-1], costs[-1]



### Cohen-Marmoret cost: Guichaoua's implication with harmonic distance
def dynamic_minimization_cohen_marmoret(chord_sequence, min_size = 8, max_size = 40, segment_size_penalty = 1, irregularity_penalty = 1, target_size = 32, persist_patterns = True, relation_type = "triad_circle"):
    """
    Dynamic programming algorithm applied to the search of the optimal segmentation of a piece of music, with Cohen-Marmoret's cost.
    
    Given the costs of all possible segments in the piece (flow), it finds the segmentation which minimizes the gloabl sum of all segment costs.
    TODO: add a reference to explain this cost.

    Parameters
    ----------
    chord_sequence : list of chords, of any type
        The chord sequence (the song) to segment.
    min_size : integer, optional
        Minimal size for a segment (except the first one and the last one). The default is 8.
    max_size : integer, optional
        Maximal size for a segment. The default is 40.
    segment_size_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the size of the segment. The default is 1.
    irregularity_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the irregularities of the polytope. The default is 1.
    target_size : integer, optional
        The optimal size, used for the penalty related to the segment size. 
        The default is 32.
    persist_patterns : boolean, optional
        A boolean, used to decide whether patterns should be computed once, and then reused (True) or not (False).
        If True, the patterns (and other informations such as antecedents of every element for instance) are persisted on a file on the machine running the code at first computation.
        (NB: I tried to store them as a variable in a huge list at first, but it resulted in errors, probably due to the size of all patterns...)
        If False, they are computed at each iteration (so for each element in each possible size).
        I strongly encourage to set it to True, as it reduces the computation time by a factor of 10.
        The default is True.

    Raises
    ------
    err
        Errors to avoid bugs at runtime (ToDebugException) or invalid arguments (InvalidArgumentValueException).

    Returns
    -------
    frontiers : list of integers
        The estimated frontiers for this segmentation.
    cost : integer
        The total cost of this segmentation.

    """
    if min_size < 2:
        raise err.InvalidArgumentValueException("Minimum size should be at least 2.")
    costs = [math.inf for i in range(len(chord_sequence))]
    costs[0] = 0
    
    symbolic_flow = []
    for chord in chord_sequence:
        if dm.is_a_chord_object(chord):
            symbolic_flow.append(chord.triad)
        else:
            symbolic_flow.append(Chord(chord).triad)

    segments_best_starts = [None for i in range(len(chord_sequence))]
    segments_best_starts[0] = 0   
    
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
            
    for current_idx in range(2, len(symbolic_flow)):
        if current_idx < min_size:
            possible_starts = [0]
        elif current_idx == len(symbolic_flow) - 1:
            possible_starts = sh.possible_segment_start(current_idx, min_size = 2, max_size = max_size)
        else:
            possible_starts = sh.possible_segment_start(current_idx, min_size = min_size, max_size = max_size)
        for possible_start_idx in possible_starts:
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")
            segment = [symbolic_flow[k] for k in range(possible_start_idx, current_idx + 1)]
            segment_size = len(segment)
            
            if persist_patterns:
                try:
                    this_bag = np.load("{}\\persisted_content\\patterns_ant_piv_for_size_{}.npy".format(CURR_DIR, segment_size), allow_pickle = True)
                except FileNotFoundError:
                    this_bag = sh.compute_patterns_with_antecedents_for_size(segment_size)
                    arr = np.array(this_bag, dtype=object)
                    np.save("{}\\persisted_content\\patterns_ant_piv_for_size_{}".format(CURR_DIR, segment_size), arr)
            else:
                this_bag = sh.compute_patterns_with_antecedents_for_size(segment_size)

            if this_bag != []:
                this_segment_cost = math.inf
                
                for a_pattern in this_bag:
                    this_polytope_cost = pc.cohen_marmoret_cost(segment, a_pattern[0], a_pattern[3], current_min = this_segment_cost, relation_type = relation_type)

                    this_polytope_cost += pc.irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = irregularity_penalty, negative_penalty = irregularity_penalty)

                    if this_polytope_cost < this_segment_cost:
                        this_segment_cost = this_polytope_cost

            else:
                warnings.warn("No Polytope is available for this size of segment. Trying sequential cost instead.")
                this_segment_cost = pc.sequential_score(segment, 0, relation_type = relation_type)
                #this_segment_cost = math.inf
            
            this_segment_cost += sh.penalty_cost_guichaoua(segment_size, target_size = target_size, positive_segment_size_penalty = segment_size_penalty, negative_segment_size_penalty = segment_size_penalty)
                
            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost < costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0
                    
            elif possible_start_idx == 1: # Maybe I should think about that bug, how 1 can be a frontier sometimes
                pass

            else:
                if costs[possible_start_idx - 1] + this_segment_cost < costs[current_idx]:
                    # Optimal cost until previous segment + cost of this segment.
                    costs[current_idx] = costs[possible_start_idx - 1] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx
                    
    frontiers = [len(chord_sequence)] #  Because frontiers are start of next segment, so it should be the chord after the next one.
    best_start_for_this_segment = segments_best_starts[len(chord_sequence) - 1]
    while best_start_for_this_segment > 0: # If best_start_for_this_segment == None, an error is raised.
        frontiers.append(best_start_for_this_segment)
        precedent_end = best_start_for_this_segment - 1 # Because previous segment ends at the chord before this one.
        best_start_for_this_segment = segments_best_starts[precedent_end]
        if precedent_end == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    frontiers.append(0) # Frontiers are here the start of a new segment, the first chord of a segment.
    return frontiers[::-1], costs[-1]#, segments_best_starts

# %% Regular polytope (if size == 2^n) or sequential cost.
def dynamic_minimization_reg_or_seq(chord_sequence, segment_size_penalty = 1, min_size = 2, max_size = 65, penalty_function = "modulo_target"):
    """
    Dynamic programming algorithm applied to the search of the optimal segmentation of a piece of music, with a simple cost.
    
    Given the costs of all possible segments in the piece (flow), it finds the segmentation which minimizes the gloabl sum of all segment costs.
    The cost is basic: uses a regular polytope if 2^n size (in Louboutin's paradigm), sequential cost otherwise.

    Parameters
    ----------
    chord_sequence : list of chords, of any type
        The chord sequence (the song) to segment.
    min_size : integer, optional
        Minimal size for a segment (except the first one and the last one). The default is 8.
    max_size : integer, optional
        Maximal size for a segment. The default is 40.
    segment_size_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the size of the segment. The default is 1.
    penalty_function : string
        Type of the penalty function, see 'penalty_cost_from_arg' of segmentation_helper.py for more details.

    Raises
    ------
    err
        Errors to avoid bugs at runtime (ToDebugException) or invalid arguments (InvalidArgumentValueException).

    Returns
    -------
    frontiers : list of integers
        The estimated frontiers for this segmentation.
    cost : integer
        The total cost of this segmentation.

    """
    raise err.OutdatedBehaviorException("Tis function doesn't work anymore actually")
    if min_size < 2:
        raise NotImplementedError("TODO better error, but minimum size should be at least 2.")
    costs = [math.inf for i in range(len(chord_sequence))]
    costs[0] = 0
    
    symbolic_flow = []
    for chord in chord_sequence:
        if dm.is_a_chord_object(chord):
            symbolic_flow.append(chord.triad)
        else:
            symbolic_flow.append(Chord(chord).triad)

    segments_best_starts = [None for i in range(len(chord_sequence))]
    segments_best_starts[0] = 0
    
    bag_of_patterns = [None]
    for i in range(1, int(math.log(max_size,2) + 1)):
        bag_of_patterns.append(pf.make_regular_polytope_pattern(i))                
            
    for current_idx in range(1, len(symbolic_flow)):
        for possible_start_idx in sh.possible_segment_start(current_idx, min_size = min_size, max_size = max_size):
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")
            segment = [symbolic_flow[k] for k in range(possible_start_idx, current_idx + 1)]
            segment_size = len(segment)
            if math.log(segment_size, 2) == int(math.log(segment_size, 2)):
                dim = int(math.log(segment_size, 2))
                this_segment_cost = pc.global_cost_computation(segment, bag_of_patterns[dim])
            else:
                this_segment_cost = pc.sequential_score(segment, 0)

            this_segment_cost += segment_size_penalty * sh.penalty_cost_from_arg(penalty_function, segment_size, target_size = 32)
                
            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost < costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0
            else:
                if costs[possible_start_idx - 1] + this_segment_cost < costs[current_idx]:
                    # Optimal cost until previous segment + cost of this segment.
                    costs[current_idx] = costs[possible_start_idx - 1] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx

    frontiers = [len(chord_sequence)]
    best_start_for_this_segment = segments_best_starts[len(chord_sequence) - 1]
    while best_start_for_this_segment > 0: # If best_start_for_this_segment == None, an error is raised.
        frontiers.append(best_start_for_this_segment)
        precedent_end = best_start_for_this_segment - 1 # Because previous segment ends at the chord before this one.
        best_start_for_this_segment = segments_best_starts[precedent_end]
        if precedent_end == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    frontiers.append(0) # Frontiers are here the start of a new segment, the first chord of a segment.
    return frontiers[::-1], costs[-1]

# %% C. Louboutin paradigm
def dynamic_minimization_louboutin(chord_sequence, min_size = 8, max_size = 40, segment_size_penalty = 1, irregularity_penalty = 1, target_size = 32, persist_patterns = True, relation_type = "triad_circle"):
    """
    Dynamic programming algorithm applied to the search of the optimal segmentation of a piece of music, with C. Louboutin's cost.
    
    Given the costs of all possible segments in the piece (flow), it finds the segmentation which minimizes the gloabl sum of all segment costs.
    TODO: add a reference to explain this cost.

    Parameters
    ----------
    chord_sequence : list of chords, of any type
        The chord sequence (the song) to segment.
    min_size : integer, optional
        Minimal size for a segment (except the first one and the last one). The default is 8.
    max_size : integer, optional
        Maximal size for a segment. The default is 40.
    segment_size_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the size of the segment. The default is 1.
    irregularity_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the irregularities of the polytope. The default is 1.
    target_size : integer, optional
        The optimal size, used for the penalty related to the segment size. 
        The default is 32.
    persist_patterns : boolean, optional
        A boolean, used to decide whether patterns should be computed once, and then reused (True) or not (False).
        If True, the patterns (and other informations such as antecedents of every element for instance) are persisted on a file on the machine running the code at first computation.
        (NB: I tried to store them as a variable in a huge list at first, but it resulted in errors, probably due to the size of all patterns...)
        If False, they are computed at each iteration (so for each element in each possible size).
        I strongly encourage to set it to True, as it reduces the computation time by a factor of 10.
        The default is True.

    Raises
    ------
    err
        Errors to avoid bugs at runtime (ToDebugException) or invalid arguments (InvalidArgumentValueException).

    Returns
    -------
    frontiers : list of integers
        The estimated frontiers for this segmentation.
    cost : integer
        The total cost of this segmentation.

    """
    if min_size < 2:
        raise err.InvalidArgumentValueException("Minimum size should be at least 2.")
    costs = [math.inf for i in range(len(chord_sequence))]
    costs[0] = 0
    
    symbolic_flow = []
    for chord in chord_sequence:
        if dm.is_a_chord_object(chord):
            symbolic_flow.append(chord.triad)
        else:
            symbolic_flow.append(Chord(chord).triad)

    segments_best_starts = [None for i in range(len(chord_sequence))]
    segments_best_starts[0] = 0   
    
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
            
    for current_idx in range(2, len(symbolic_flow)):
        if current_idx < min_size:
            possible_starts = [0]
        elif current_idx == len(symbolic_flow) - 1:
            possible_starts = sh.possible_segment_start(current_idx, min_size = 2, max_size = max_size)
        else:
            possible_starts = sh.possible_segment_start(current_idx, min_size = min_size, max_size = max_size)
        for possible_start_idx in possible_starts:
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")
            segment = [symbolic_flow[k] for k in range(possible_start_idx, current_idx + 1)]
            segment_size = len(segment)
            
            if persist_patterns:
                try:
                    this_bag = np.load("{}\\persisted_content\\compute_patterns_and_ppp_for_size_{}.npy".format(CURR_DIR, segment_size), allow_pickle = True)
                except FileNotFoundError:
                    this_bag = sh.compute_patterns_and_ppp_for_size(segment_size)
                    arr = np.array(this_bag, dtype=object)
                    np.save("{}\\persisted_content\\compute_patterns_and_ppp_for_size_{}".format(CURR_DIR, segment_size), arr)
            else:
                this_bag = sh.compute_patterns_and_ppp_for_size(segment_size)

            if this_bag != []:
                this_segment_cost = math.inf
                
                for a_pattern in this_bag:
                    this_polytope_cost = math.inf
                    for i in range(len(a_pattern[0])):
                        this_ppp_cost = pc.louboutin_cost_for_a_ppp(segment, a_pattern[0][i], a_pattern[3][i], a_pattern[4][i], current_min = this_segment_cost, relation_type = relation_type)
                        if this_ppp_cost < this_polytope_cost:
                            this_polytope_cost = this_ppp_cost
                    
                    this_polytope_cost += pc.irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = irregularity_penalty, negative_penalty = irregularity_penalty)

                    if this_polytope_cost < this_segment_cost:
                        this_segment_cost = this_polytope_cost

            else:
                warnings.warn("No Polytope is available for this size of segment. Trying sequential cost instead.")
                this_segment_cost = pc.sequential_score(segment, 0, relation_type = relation_type)
                #this_segment_cost = math.inf
            
            this_segment_cost += sh.penalty_cost_guichaoua(segment_size, target_size = target_size, positive_segment_size_penalty = segment_size_penalty, negative_segment_size_penalty = segment_size_penalty)
                
            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost < costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0

            else:
                if costs[possible_start_idx - 1] + this_segment_cost < costs[current_idx]:
                    # Optimal cost until previous segment + cost of this segment.
                    costs[current_idx] = costs[possible_start_idx - 1] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx
                    
    frontiers = [len(chord_sequence)] #  Because frontiers are start of next segment, so it should be the chord after the next one.
    best_start_for_this_segment = segments_best_starts[len(chord_sequence) - 1]
    while best_start_for_this_segment > 0: # If best_start_for_this_segment == None, an error is raised.
        frontiers.append(best_start_for_this_segment)
        precedent_end = best_start_for_this_segment - 1 # Because previous segment ends at the chord before this one.
        best_start_for_this_segment = segments_best_starts[precedent_end]
        if precedent_end == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    frontiers.append(0) # Frontiers are here the start of a new segment, the first chord of a segment.
    return frontiers[::-1], costs[-1]

### Mix of PPP and implication: Louboutaoua
def dynamic_minimization_louboutaoua(chord_sequence, min_size = 8, max_size = 40, segment_size_penalty = 1, irregularity_penalty = 1, target_size = 32, persist_patterns = True, relation_type = "triad_circle"):
    """
    Dynamic programming algorithm applied to the search of the optimal segmentation of a piece of music, with Louboutaoua's cost.
    
    Given the costs of all possible segments in the piece (flow), it finds the segmentation which minimizes the gloabl sum of all segment costs.
    TODO: add a reference to explain this cost.

    Parameters
    ----------
    chord_sequence : list of chords, of any type
        The chord sequence (the song) to segment.
    min_size : integer, optional
        Minimal size for a segment (except the first one and the last one). The default is 8.
    max_size : integer, optional
        Maximal size for a segment. The default is 40.
    segment_size_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the size of the segment. The default is 1.
    irregularity_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the irregularities of the polytope. The default is 1.
    target_size : integer, optional
        The optimal size, used for the penalty related to the segment size. 
        The default is 32.
    persist_patterns : boolean, optional
        A boolean, used to decide whether patterns should be computed once, and then reused (True) or not (False).
        If True, the patterns (and other informations such as antecedents of every element for instance) are persisted on a file on the machine running the code at first computation.
        (NB: I tried to store them as a variable in a huge list at first, but it resulted in errors, probably due to the size of all patterns...)
        If False, they are computed at each iteration (so for each element in each possible size).
        I strongly encourage to set it to True, as it reduces the computation time by a factor of 10.
        The default is True.

    Raises
    ------
    err
        Errors to avoid bugs at runtime (ToDebugException) or invalid arguments (InvalidArgumentValueException).

    Returns
    -------
    frontiers : list of integers
        The estimated frontiers for this segmentation.
    cost : integer
        The total cost of this segmentation.

    """
    if min_size < 2:
        raise err.InvalidArgumentValueException("Minimum size should be at least 2.")
    costs = [math.inf for i in range(len(chord_sequence))]
    costs[0] = 0
    
    symbolic_flow = []
    for chord in chord_sequence:
        if dm.is_a_chord_object(chord):
            symbolic_flow.append(chord.triad)
        else:
            symbolic_flow.append(Chord(chord).triad)

    segments_best_starts = [None for i in range(len(chord_sequence))]
    segments_best_starts[0] = 0   
                
    for current_idx in range(2, len(symbolic_flow)):
        if current_idx < min_size:
            possible_starts = [0]
        elif current_idx == len(symbolic_flow) - 1:
            possible_starts = sh.possible_segment_start(current_idx, min_size = 2, max_size = max_size)
        else:
            possible_starts = sh.possible_segment_start(current_idx, min_size = min_size, max_size = max_size)
        for possible_start_idx in possible_starts:
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")
            segment = [symbolic_flow[k] for k in range(possible_start_idx, current_idx + 1)]
            segment_size = len(segment)
            
            if persist_patterns:
                try:
                    this_bag = np.load("{}\\persisted_content\\patterns_and_ppps_with_antecedents_for_size_{}.npy".format(CURR_DIR, segment_size), allow_pickle = True)
                except FileNotFoundError:
                    this_bag = sh.compute_patterns_with_ppp_and_antecedents_for_size(segment_size)
                    arr = np.array(this_bag, dtype=object)
                    np.save("{}\\persisted_content\\patterns_and_ppps_with_antecedents_for_size_{}".format(CURR_DIR, segment_size), arr)
            else:
                this_bag = sh.compute_patterns_with_ppp_and_antecedents_for_size(segment_size)

            if this_bag != []:
                this_segment_cost = math.inf
                
                for a_pattern in this_bag:
                    this_polytope_cost = math.inf
                    for i in range(len(a_pattern[0])):
                        this_ppp_cost = pc.louboutaoua_cost(segment, a_pattern[0][i], a_pattern[3], a_pattern[4], a_pattern[5], a_pattern[6][i], current_min = this_polytope_cost, relation_type = relation_type)
                        if this_ppp_cost < this_polytope_cost:
                            this_polytope_cost = this_ppp_cost
                    
                    this_polytope_cost += pc.irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = irregularity_penalty, negative_penalty = irregularity_penalty)

                    if this_polytope_cost < this_segment_cost:
                        this_segment_cost = this_polytope_cost

            else:
                warnings.warn("No Polytope is available for this size of segment. Trying sequential cost instead.")
                this_segment_cost = pc.sequential_score(segment, 0, relation_type = relation_type)
                #this_segment_cost = math.inf
            
            this_segment_cost += sh.penalty_cost_guichaoua(segment_size, target_size = target_size, positive_segment_size_penalty = segment_size_penalty, negative_segment_size_penalty = segment_size_penalty)
                
            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost < costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0

            else:
                if costs[possible_start_idx - 1] + this_segment_cost < costs[current_idx]:
                    # Optimal cost until previous segment + cost of this segment.
                    costs[current_idx] = costs[possible_start_idx - 1] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx
                    
    frontiers = [len(chord_sequence)] #  Because frontiers are start of next segment, so it should be the chord after the next one.
    best_start_for_this_segment = segments_best_starts[len(chord_sequence) - 1]
    while best_start_for_this_segment > 0: # If best_start_for_this_segment == None, an error is raised.
        frontiers.append(best_start_for_this_segment)
        precedent_end = best_start_for_this_segment - 1 # Because previous segment ends at the chord before this one.
        best_start_for_this_segment = segments_best_starts[precedent_end]
        if precedent_end == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    frontiers.append(0) # Frontiers are here the start of a new segment, the first chord of a segment.
    return frontiers[::-1], costs[-1]      

# def new_viterbi_loop_size(chord_sequence, positive_penalty = 2.25, negative_penalty = 3, min_size = 8, max_size = 40, positive_segment_size_penalty = 0, negative_segment_size_penalty = 0.125, target_size = 32):
#     # Work in progress, not functionning right now, but idea is to computes score by the size of the segments first, and then find the optimal segmentation.
#     # It would avoid to reload/recompute patterns every time.
#     warnings.warn("This function (new_viterbi_loop_size) does not work, work still in progress.")
#     if min_size < 2:
#         raise NotImplementedError("TODO better error, but minimum size should be at least 2.")
#     costs = [math.inf for i in range(len(chord_sequence))]
#     costs[0] = 0
    
#     symbolic_flow = []
#     for chord in chord_sequence:
#         if dm.is_a_chord_object(chord):
#             symbolic_flow.append(chord.triad)
#         else:
#             symbolic_flow.append(Chord(chord).triad)

#     segments_best_starts = [None for i in range(len(chord_sequence))]
#     segments_best_starts[0] = 0
#     song_size = len(chord_sequence)
#     final_matrix = np.full((song_size, song_size), math.inf)
    
#     CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    
#     for current_size in range(min_size, max_size + 1):
#         bag_of_patterns = compute_patterns_with_antecedents_for_size(current_size)
#         if bag_of_patterns != []:
#             for current_end in range(current_size, song_size):
#                 possible_start = current_end - current_size
#                 segment = [symbolic_flow[k] for k in range(possible_start, current_end + 1)]
        
#                 this_segment_cost = math.inf
#                 for a_pattern in bag_of_patterns:
#                     this_polytope_cost = guichaoua_cost(segment, a_pattern[0], a_pattern[3], a_pattern[4], a_pattern[5])
#                     this_polytope_cost += irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = positive_penalty, negative_penalty = negative_penalty)
#                     if this_polytope_cost < this_segment_cost:
#                             this_segment_cost = this_polytope_cost
#                 this_segment_cost += penalty_cost_guichaoua(current_size, target_size = target_size, positive_segment_size_penalty = positive_segment_size_penalty, negative_segment_size_penalty = negative_segment_size_penalty)
#                 final_matrix[possible_start][current_end] = this_segment_cost
            
#     for current_idx in range(2, len(symbolic_flow)):
#         if current_idx < min_size:
#             bag_of_patterns = compute_patterns_with_antecedents_for_size(current_idx + 1)
#             if bag_of_patterns != []:
#                 segment = [symbolic_flow[k] for k in range(0, current_idx + 1)]
#                 this_segment_cost = math.inf
#                 for a_pattern in bag_of_patterns:
#                     this_polytope_cost = guichaoua_cost(segment, a_pattern[0], a_pattern[3], a_pattern[4], a_pattern[5])
#                     this_polytope_cost += irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = positive_penalty, negative_penalty = negative_penalty)
#                     if this_polytope_cost < this_segment_cost:
#                             this_segment_cost = this_polytope_cost
#                 this_segment_cost += penalty_cost_guichaoua(current_size, target_size = target_size, positive_segment_size_penalty = positive_segment_size_penalty, negative_segment_size_penalty = negative_segment_size_penalty)
#                 final_matrix[0][current_idx - 1] = this_segment_cost
                
#         # elif current_idx == song_size - 1:
#         #     for possible_start in range(song_size - min_size, song_size - 2):
#         #         seg_size = song_size - 1 - possible_start
#         #         bag_of_patterns = compute_patterns_with_antecedents_for_size(seg_size)
#         #         if bag_of_patterns != []:
#         #             segment = [symbolic_flow[k] for k in range(possible_start, song_size)]
#         #             this_segment_cost = math.inf
#         #             for a_pattern in bag_of_patterns:
#         #                 this_polytope_cost = guichaoua_cost(segment, a_pattern[0], a_pattern[3], a_pattern[4], a_pattern[5])
#         #                 this_polytope_cost += irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = positive_penalty, negative_penalty = negative_penalty)
#         #                 if this_polytope_cost < this_segment_cost:
#         #                         this_segment_cost = this_polytope_cost
#         #             this_segment_cost += penalty_cost_guichaoua(current_size, target_size = target_size, positive_segment_size_penalty = positive_segment_size_penalty, negative_segment_size_penalty = negative_segment_size_penalty)
#         #             final_matrix[possible_start][song_size - 1] = this_segment_cost
                
#         else:
#             for possible_start_idx in seq_seg.possible_segment_start(current_idx, min_size = min_size, max_size = max_size):
#                 if possible_start_idx < 0:
#                     raise err.ToDebugException("Invalid value of start index.")
                    
#                 this_segment_cost = final_matrix[possible_start_idx][current_idx]
            
#                 # Avoiding errors, as segment_cost are initially set to -inf.
#                 if possible_start_idx == 0:
#                     if this_segment_cost < costs[current_idx]:
#                         costs[current_idx] = this_segment_cost
#                         segments_best_starts[current_idx] = 0
#                 else:
#                     if costs[possible_start_idx - 1] + this_segment_cost < costs[current_idx]:
#                         # Optimal cost until previous segment + cost of this segment.
#                         costs[current_idx] = costs[possible_start_idx - 1] + this_segment_cost
#                         segments_best_starts[current_idx] = possible_start_idx

#     frontiers = [len(chord_sequence) - 1]
#     best_start_for_this_segment = segments_best_starts[len(chord_sequence) - 1]
#     while best_start_for_this_segment > 0: # If best_start_for_this_segment == None, an error is raised.
#         frontiers.append(best_start_for_this_segment)
#         precedent_end = best_start_for_this_segment - 1 # Because previous segment ends at the chord before this one.
#         best_start_for_this_segment = segments_best_starts[precedent_end]
#         if precedent_end == None:
#             raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
#     frontiers.append(0) # Frontiers are here the start of a new segment, the first chord of a segment.
#     return frontiers[::-1], costs[-1]        

