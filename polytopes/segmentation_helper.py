# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:07:30 2019

@author: amarmore
"""
import math

from polytopes.model.chord import Chord
import polytopes.data_manipulation as dm
import polytopes.polytopical_costs as pc
import polytopes.chord_movement as mvt
import polytopes.pattern_manip as pm
import polytopes.pattern_factory as pf
import polytopes.model.errors as err
import numpy as np
import time

def possible_segment_start(idx, min_size = 2, max_size = None):
    """
    Generate the list of all possible starts of segments given the index of its end.
    
    Parameters
    ----------
    idx: integer
        The end of a segment.
    min_size: integer
        Minimal length of a segment.
    max_size: integer
        Maximal length of a segment.
        
    Returns
    -------
    list of integers
        All potentials starts of structural segments.
    """
    if min_size < 1: # No segment should be allowed to be 0 size
        min_size = 1
    if max_size == None:
        return range(0, idx - min_size + 2)
    else:
        if idx >= max_size:
            return range(idx - max_size + 1, idx - min_size + 2)
        elif idx >= min_size:
            return range(0, idx - min_size + 2)
        else:
            return []

def penalty_cost_from_arg(penalty_func, segment_length, target_size = 16):
    """
    Returns a penalty cost, function of the size of the segment.
    The penalty function has to be specified, and is bound to evolve in the near future,
    so this docstring won't explain it.
    Instead, you'll have to read the code, sorry! It is pretty straightforward though.

    Parameters
    ----------
    penalty_func : string
        Identifier of the penalty function.
    segment_length : integer
        Size of the segment.

    Returns
    -------
    float
        The penalty cost.

    """
    if penalty_func == "modulo4":        
        # if segment_length % 4 != 0:
        #     return 1/(min(segment_length % 4, -segment_length % 4))
        # else:
        #     return 0
        if segment_length %16 == 0: # 4/4 meter, so 4 bears of 4 beats
            return 0
        elif segment_length %8 == 0: # 4/4 meter, si odd nb of bars of 4 beats
            return 1/2
        else:
            return 1
    if penalty_func == "modulo_target":        
        if segment_length == target_size:
            return 0
        elif segment_length %target_size == 0:
            return 1/4
        elif segment_length %target_size == 0: # Subdividing the target_size
            return 1/2
        else:
            return 1

    if penalty_func == "sargentdemi": 
         return abs(segment_length - target_size) ** (1/2)
    if penalty_func == "sargentun": 
         return abs(segment_length - target_size)
    if penalty_func == "sargentdeux": 
         return abs(segment_length - target_size) ** 2
    if penalty_func == "modulo8": 
        raise err.InvalidArgumentValueException("This penalty function don't exist anymore, maybe you mean 'modulo_target'?.")
    else:
        raise err.InvalidArgumentValueException("Penalty function not understood.")

def penalty_cost_guichaoua(segment_length, target_size = 32, positive_segment_size_penalty = 0, negative_segment_size_penalty = 0.125):
    """
    Penalty cost function (for the size of the segment) as defined by C. Guichaoua in his thesis.
    
    In fact, it's the same as "sargentun" above (i.e. difference between actual size and a target_size), but with different parameters for exceeding or subceeding.

    Parameters
    ----------
    segment_length : integer
        Size of the segment.
    target_size : integer, optional
        Optimal size. The default is 32.
    positive_segment_size_penalty : float, optional
        penalty when the segment_size exceeds the optimal/target size. The default is 0.
    negative_segment_size_penalty : TYPE, optional
        penalty when the segment_size subceeds the optimal/target size. The default is 0.125.

    Returns
    -------
    float
        The penalty, to add to the score.

    """
    if segment_length >= target_size:
        return positive_segment_size_penalty * (segment_length - target_size)
    else:
        return negative_segment_size_penalty * (target_size - segment_length)
    
def update_correct_antecedents(current_elt, suc_ant_piv, suc_corr_ant):
    """
    Updating the valid antecedents of one element's successors. USed in C. Guichaoua's paradigm.

    Parameters
    ----------
    current_elt : integer
        The current element, which successors are to update.
    suc_ant_piv : list of tuples of integers
        The antecedents and pivots for this successor. of 'current_elt'.
    suc_corr_ant : list of integers
        Valid antecedents (updated) of this successor of 'current_elt'.
        
    Raises
    ------
    ToDebugException
        When the pivot is null, which is not a normal behavior.

    Returns
    -------
    new_correct_antecedents : list of integers (or void list)
        The updated antecedents for this successors of the current element.

    """
    # Find the pivot of elt relatively to this_elt_successor
    pivot = None
    for couple in suc_ant_piv:
        if couple[0] == current_elt:
            pivot = couple[1]
    if pivot == None:
        raise err.ToDebugException("Pivot shouldn't be null: Element: {}, (ant,piv) for successors: {}.".format(current_elt, suc_ant_piv))
        
    # Keep only the current element or its successors as future valid antecedents for the current element' successors
    new_correct_antecedents = []
    for ant_to_update in suc_corr_ant:
        if ant_to_update == current_elt or ant_to_update == pivot:
            new_correct_antecedents.append(ant_to_update)
    return new_correct_antecedents


# %% Pattern and antecedents/successors loaders, to compute them ince and for all.
def compute_patterns_with_ppp_and_antecedents_for_size(size):
    """
    Compute patterns of a given size and information related to this pattern, to avoid several computation in dynamic programming algorithm.
    
    This function works for Louboutaoua cost, and computes:
        - all_ppps (all ppps of a given pattern, a list)
        - add (adding_code)
        - dele (deleting_code)
        - antecedents_and_pivots (couple ant,piv for all elements of this pattern)
        - successors (successors for every element)
        - correct_antecedents (initial valid antecedent for every element)
        - bag_of_direct_antecedents (direct antecedent for every element of every ppp (so list of list of integers))

    Parameters
    ----------
    size : integer
        The desired size of patterns.

    Returns
    -------
    list
        List of all the elements listed above.

    """
    to_return = []
    codes = pf.get_unique_codes(size)
    if codes == []:
        return []
    for add, dele in codes:
        local_pattern = pf.make_indexed_pattern(int(round(math.log(size,2))), adding_code = add, deleting_code = dele)
        all_ppps = pm.generate_ppp(local_pattern)
        bag_of_direct_antecedents = []
        for a_ppp in all_ppps:
            bag_of_direct_antecedents.append(pm.compute_direct_antecedents(a_ppp))

        antecedents_and_pivots = [None]
        successors = [[]]
        correct_antecedents = [None]
        for elt in range(1, pf.get_pattern_size(local_pattern)):
            elt_idx = pm.get_index_from_element(elt, local_pattern)
            antecedents_and_pivots.append(pm.get_antecedents_with_pivots_from_index(elt_idx, local_pattern))
            this_correct_antecedents = []
            for ant in pm.get_antecedents_from_index(elt_idx, local_pattern):
                if ant != 0:
                    this_correct_antecedents.append(ant)
            correct_antecedents.append(this_correct_antecedents)
            successors.append(pm.get_successors_from_index(elt_idx, local_pattern))
        to_return.append([all_ppps, add, dele, antecedents_and_pivots, successors, correct_antecedents, bag_of_direct_antecedents])
    return to_return


def compute_patterns_with_antecedents_for_size(size):
    """
    Compute patterns of a given size and information related to this pattern, to avoid several computation in dynamic programming algorithm.
    
    This function works for Guichaoua and Cohen-Marmoret costs, and computes:
        - local_pattern (a given pattern)
        - add (adding_code)
        - dele (deleting_code)
        - antecedents_and_pivots (couple ant,piv for all elements of this pattern)
        - successors (successors for every element)
        - correct_antecedents (initial valid antecedent for every element)

    Parameters
    ----------
    size : integer
        The desired size of patterns.

    Returns
    -------
    list
        List of all the elements listed above.

    """
    to_return = []
    codes = pf.get_unique_codes(size)
    if codes == []:
        return []
    for add, dele in codes:
        local_pattern = pf.make_indexed_pattern(int(round(math.log(size,2))), adding_code = add, deleting_code = dele)
        antecedents_and_pivots = [None]
        successors = [[]]
        correct_antecedents = [None]
        for elt in range(1, pf.get_pattern_size(local_pattern)):
            elt_idx = pm.get_index_from_element(elt, local_pattern)
            antecedents_and_pivots.append(pm.get_antecedents_with_pivots_from_index(elt_idx, local_pattern))
            this_correct_antecedents = []
            for ant in pm.get_antecedents_from_index(elt_idx, local_pattern):
                if ant != 0:
                    this_correct_antecedents.append(ant)
            correct_antecedents.append(this_correct_antecedents)
            successors.append(pm.get_successors_from_index(elt_idx, local_pattern))
        to_return.append([local_pattern, add, dele, antecedents_and_pivots, successors, correct_antecedents])
    return to_return


def compute_patterns_with_global_antecedents_for_size(size):
    """
    Compute patterns of a given size and information related to this pattern, to avoid several computation in dynamic programming algorithm.
    
    This function works for Guichaoua cost, with global antecedents, and computes:
        - local_pattern (a given pattern)
        - add (adding_code)
        - dele (deleting_code)
        - antecedents_and_pivots (couple ant,piv for all elements of this pattern, but they are global antecedents)
        - successors (successors for every element, but they are global successors)
        - correct_antecedents (initial valid antecedent for every element)

    Parameters
    ----------
    size : integer
        The desired size of patterns.

    Returns
    -------
    list
        List of all the elements listed above.

    """
    to_return = []
    codes = pf.get_unique_codes(size)
    if codes == []:
        return []
    for add, dele in codes:
        local_pattern = pf.make_indexed_pattern(int(round(math.log(size,2))), adding_code = add, deleting_code = dele)
        antecedents_and_pivots = [None]
        successors = [[]]
        correct_antecedents = [None]
        for elt in range(1, pf.get_pattern_size(local_pattern)):
            elt_idx = pm.get_index_from_element(elt, local_pattern)
            antecedents_and_pivots.append(pm.get_global_antecedents_with_pivots_from_index(elt_idx, local_pattern))
            this_correct_antecedents = []
            for ant, _ in antecedents_and_pivots[-1]:
                if ant != 0:
                    this_correct_antecedents.append(ant)
            correct_antecedents.append(this_correct_antecedents)
            successors.append(pm.get_global_successors_from_index(elt_idx, local_pattern))
        to_return.append([local_pattern, add, dele, antecedents_and_pivots, successors, correct_antecedents])
    return to_return

def compute_patterns_and_ppp_for_size(size):
    """
    Compute patterns of a given size and information related to this pattern, to avoid several computation in dynamic programming algorithm.
    
    This function works for Louboutin cost, and computes:
        - all_ppps (all ppps of a given pattern, a list)
        - add (adding_code)
        - dele (deleting_code)
        - patterns_of_one (all_ppps of before, but as patterns of ones, so list of patterns of ones)
        - reindex (only the indexed of all_ppps (flattening each ppp to keep only the elements), so a list of list of integers)

    Parameters
    ----------
    size : integer
        The desired size of patterns.

    Returns
    -------
    list
        List of all the elements listed above.

    """
    to_return = []
    codes = pf.get_unique_codes(size)
    if codes == []:
        return []
    for add, dele in codes:
        local_pattern = pf.make_indexed_pattern(int(round(math.log(size,2))), adding_code = add, deleting_code = dele)
        all_ppps = pm.generate_ppp(local_pattern)
        patterns_of_one = []
        reindex = []
        for a_ppp in all_ppps:
            patterns_of_one.append(pf.extract_pattern_from_indexed_pattern(a_ppp))
            reindex.append(pf.flatten_pattern(a_ppp))
        to_return.append([all_ppps, add, dele, patterns_of_one, reindex])
    return to_return

