# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:11:11 2019

@author: amarmore

File containing cost function for polytopes, in the different paradigms.
In general, cost functions are defined to be computed in a loop (dynamic minimization algorithm),
and are coded such that they don't have to recompute pattern at each iteration.
In that sense, they're not defined (or at least, made user-friendly) for computing a cost on a single pattern.
TODO: maybe define such functions.
"""

import math
import numpy as np
from numba import jit

import polytopes.segmentation_helper as sh
import polytopes.chord_movement as mvt
import polytopes.pattern_manip as pm
import polytopes.pattern_factory as pf
import polytopes.triad_transformations as tt
import polytopes.model.errors as err
import polytopes.accelerated_polytopical_costs as acc_pc

# %% Louboutin (and more generally, high-level S&C) pardigm
def louboutin_cost_for_a_ppp(segment, a_ppp, pattern_of_ones, reindex, current_min = math.inf, relation_type = "triad_circle"):
    """
    Compute the cost in the Louboutin's paradigm, for this segment and this PPP 'a_ppp'.

    Parameters
    ----------
    segment : list of Chord, in any form
        The segment, on which to compute the score.
    a_ppp : nested list of integers
        The PPP.
    pattern_of_ones : nested list of ones
        The pattern of the PPP, but only made of one (for computing the s_and_c cost).
    reindex : list of integers
        Order of the elements in the PPP, in order to reindex the segment.
        It consists in the PPP, flattened.
    current_min : integer, optional
        The current minimal cost.
        It's an acceleration parameter, to check that the current cost we're computing' isn't already higher than the current minimal value.
        If it's the case, this ppp won't be chosen, so we don't need an exact score, and it returns math.inf.
        The default is math.inf.

    Returns
    -------
    integer
        The cost of this segment on this ppp in the Louboutin paradigm.

    """
    # Old behavior, working only with the PPP (could be useful)
    # reindexed_segment = pm.swap_chord_sequence(segment, pm.flatten_pattern(a_ppp))
    # pattern_made_of_ones = pf.extract_pattern_from_indexed_pattern(a_ppp)
    
    new_segment = pm.swap_chord_sequence(segment, reindex)
    return polytopic_scale_s_and_c_cost_computation(new_segment, pattern_of_ones, current_min = current_min, relation_type = relation_type)

def polytopic_scale_s_and_c_cost_computation(symbol_flow, polytope_pattern, extended_s_and_c = False, current_min = math.inf, relation_type = "triad_circle"):
    """
    Compute the cost of a chord sequence on a polytope, in the Louboutin paradigm (extended by A. Marmoret to irregular polytopes).

    Parameters
    ----------
    symbol_flow : list of Chord, in any form
        The segment, on which to compute the score.
    polytope_pattern : nested list of 1
        The pattern of oness corresponding to the polytope on which to compute the score.
    extended_s_and_c : boolean, optional
        A mode of computing scores with other types of S&Cs.
        Only used in tests, which were not interesting enough.
        TODO: Should I keep this score computation mode?
        The default is False.
    current_min : integer, optional
        The current minimal cost.
        It's an acceleration parameter, to check that the current cost we're computing' isn't already higher than the current minimal value.
        If it's the case, this ppp won't be chosen, so we don't need an exact score, and it returns math.inf.
        The default is math.inf.

    Returns
    -------
    integer
        The cost of this segment on this polytope in the Louboutin paradigm.

    """
    # Global cost means: low systems cost, and computing the cost of the primers.
    if pf.get_pattern_dimension(polytope_pattern) <= 2:
        return recursive_low_level_splitter_for_s_and_c_cost(symbol_flow, polytope_pattern, extended_s_and_c = extended_s_and_c, relation_type = relation_type)
    else:
        inner_systems_cost = recursive_low_level_splitter_for_s_and_c_cost(symbol_flow, polytope_pattern, extended_s_and_c = extended_s_and_c, relation_type = relation_type)
        if inner_systems_cost > current_min:
            # Cost is already higher than current min, so avoid further computation.
            return math.inf
        primers_chords = recursively_find_primers_chords(symbol_flow, polytope_pattern)
        primer_pattern = extract_pattern_from_primers_chords(primers_chords)
        primer_symbol_flow = pf.flatten_pattern(primers_chords)
        return inner_systems_cost + polytopic_scale_s_and_c_cost_computation(primer_symbol_flow, primer_pattern, extended_s_and_c = extended_s_and_c, relation_type = relation_type)

def recursive_low_level_splitter_for_s_and_c_cost(symbol_flow, polytope_pattern, extended_s_and_c = False, relation_type = "triad_circle"):
    """
    Split the polytope in a list of dimension 2 polytopes, and compute score on these polytopes, as defined in the Louboutin paradigm, extended by A. Marmoret.
    
    TODO: insert a ref when available here, because it reaaaaally needs one.

    Parameters
    ----------
    symbol_flow : list of Chord, in any form
        The segment, on which to compute the score.
    polytope_pattern : nested list of 1
        The pattern of oness corresponding to the polytope on which to compute the score.
    extended_s_and_c : boolean, optional
        A mode of computing scores with other types of S&Cs.
        Only used in tests, which were not interesting enough.
        TODO: Should I keep this score computation mode?
        The default is False.

    Raises
    ------
    PatternAndSequenceIncompatible
        Error raised when the pattern and the sequence are of different sizes.

    Returns
    -------
    integer (but could be float with other cost functions)
        The sum of the costs of low-level systems (all dimension 2 polytopes).

    """
    # Computing the cost of elements as the sum of all dim 2 polytopes (excluding primers).
    if len(symbol_flow) != pf.get_pattern_size(polytope_pattern):
        raise err.PatternAndSequenceIncompatible("Chord sequence and pattern are of different lengths.")
    if pf.get_pattern_dimension(polytope_pattern) <= 2:
        #print(polytope_pattern.pattern)
        #print(polytope.get_list_of_symbols())
        if extended_s_and_c:
            return dim_two_extended_s_and_c_cost(symbol_flow, polytope_pattern)
        else:
            return low_level_system_s_and_c_cost(symbol_flow, polytope_pattern, relation_type = relation_type)
    else:
        if len(polytope_pattern) == 1:
            first_nested_pattern = polytope_pattern[0]
            return recursive_low_level_splitter_for_s_and_c_cost(symbol_flow, first_nested_pattern, extended_s_and_c = extended_s_and_c, relation_type = relation_type)
        else:
            first_nested_pattern = polytope_pattern[0]
            cost_first_nested_pattern = recursive_low_level_splitter_for_s_and_c_cost(symbol_flow[:pf.get_pattern_size(first_nested_pattern)], first_nested_pattern, extended_s_and_c = extended_s_and_c, relation_type = relation_type)
            second_nested_pattern = polytope_pattern[1]
            cost_second_nested_pattern = recursive_low_level_splitter_for_s_and_c_cost(symbol_flow[pf.get_pattern_size(first_nested_pattern):], second_nested_pattern, extended_s_and_c = extended_s_and_c, relation_type = relation_type)
            return cost_first_nested_pattern + cost_second_nested_pattern

def low_level_system_s_and_c_cost(symbol_flow, polytope_pattern, relation_type = "triad_circle"):
    """
    Compute the cost of a low-level system (dimension 2 polytope).

    Parameters
    ----------
    symbol_flow : list of Chord, in any form
        The segment, on which to compute the score.
    polytope_pattern : nested list of 1
        The pattern of oness corresponding to the polytope on which to compute the score.

    Raises
    ------
    PatternAndSequenceIncompatible
        Error raised when the pattern and the sequence are of different sizes.

    Returns
    -------
    integer (but could be float with other cost functions)
        Cost of this low-level system (dimension 2 polytope).

    """
    # Computing the cost of a dim 2 polytopes.
    pattern_size = pf.get_pattern_size(polytope_pattern)
    if len(symbol_flow) != pattern_size:
        raise err.PatternAndSequenceIncompatible("The pattern's length is different than the the chord sequence's length, which make them incompatible.") from None
    if pattern_size < 2:
        raise err.UnexpectedDim1Pattern("Side effect (pattern of size 1), should it happen ?")
    if pf.get_pattern_dimension(polytope_pattern) > 2:
        raise err.UnexpectedDimensionForPattern("This pattern is of high dimension (higher than 2), can't compute a cost on it.")

    if pattern_size == 2:
        return score_relation_switcher(relation_type, symbol_flow[0], symbol_flow[1])
    s_and_c = []
    primer = symbol_flow[0]
    score = 0
    sequence_idx = 0
    for one_dimension_pattern in polytope_pattern:
        if one_dimension_pattern == [1,1]:
            s_and_c.append(symbol_flow[sequence_idx])
            s_and_c.append(symbol_flow[sequence_idx + 1])
            sequence_idx += 2
            
        elif one_dimension_pattern == [1,(1,1)]:
            score += score_relation_switcher(relation_type, symbol_flow[sequence_idx + 1], symbol_flow[sequence_idx + 2])
            s_and_c.append(symbol_flow[sequence_idx])
            s_and_c.append(symbol_flow[sequence_idx + 1])
            sequence_idx += 3
            
        elif one_dimension_pattern == [(1,1),(1,1)]:
            score += score_relation_switcher(relation_type, symbol_flow[sequence_idx],symbol_flow[sequence_idx + 1])
            score += score_relation_switcher(relation_type, symbol_flow[sequence_idx + 2],symbol_flow[sequence_idx + 3])
            s_and_c.append(symbol_flow[sequence_idx])
            s_and_c.append(symbol_flow[sequence_idx + 2])
            sequence_idx += 4
        
        elif one_dimension_pattern == [1]:
            if sequence_idx != 0:
                score += score_relation_switcher(relation_type, primer, symbol_flow[sequence_idx])
            sequence_idx += 1

        elif one_dimension_pattern == [(1,1)]:
            if sequence_idx != 0:
                score += score_relation_switcher(relation_type, primer, symbol_flow[sequence_idx])
            score += score_relation_switcher(relation_type, symbol_flow[sequence_idx], symbol_flow[sequence_idx + 1])
            sequence_idx += 2

        else:
            raise err.PatternToDebugError("Unknown pattern: {}".format(one_dimension_pattern))
            
    if len(s_and_c) == 4:
        score += s_and_c_cost(s_and_c)

    elif len(s_and_c) == 2:
        score += score_relation_switcher(relation_type, s_and_c[0], s_and_c[1])
    
    elif len(s_and_c) != 0:
        raise err.PatternToDebugError("Pattern resulting in {}-element S&C, to debug ({})".format(len(s_and_c), str(polytope_pattern)))
    
    return score
    
def dim_two_extended_s_and_c_cost(symbol_flow, polytope_pattern):
    raise err.OutdatedBehaviorException("Extended S&C cost isn't supported anymore.")
#     """
#     Compute the cost of a low-level system (dimension 2 polytope), but in a new paradigm.
    
#     This cost function is not promising enough, so it's kind of left here for posterity.
#     TODO: maybe delete it (at least think about it)

#     Parameters
#     ----------
#     symbol_flow : list of Chord, in any form
#         The segment, on which to compute the score.
#     polytope_pattern : nested list of 1
#         The pattern of oness corresponding to the polytope on which to compute the score.

#     Raises
#     ------
#     PatternAndSequenceIncompatible
#         Error raised when the pattern and the sequence are of different sizes.

#     Returns
#     -------
#     integer (but could be float with other cost functions)
#         Cost of this low-level system (dimension 2 polytope).

#     """
#     # Computing the cost of a dim 2 polytopes in the extended s&c scheme.
#     pattern_size = pf.get_pattern_size(polytope_pattern)
#     if len(symbol_flow) != pattern_size:
#         raise err.PatternAndSequenceIncompatible("The pattern's length is different than the the chord sequence's length, which make them incompatible.") from None
#     if pattern_size < 2:
#         raise err.UnexpectedDim1Pattern("Side effect (pattern of size 1), should it happen ?")
#     if pf.get_pattern_dimension(polytope_pattern) > 2:
#         raise err.UnexpectedDimensionForPattern("This pattern is of high dimension (higher than 2), can't compute a cost on it.")

#     if pattern_size == 2:
#         return voice_leading_cost(symbol_flow[0], symbol_flow[1])
    
#     if pf.get_pattern_dimension(polytope_pattern) != 2:
#         raise err.PatternToDebugError("Should be of dimension 2, but is {}" + str(polytope_pattern))

#     if pattern_size == 3:
#         if polytope_pattern == [[1,1],[1]]:
#             return voice_leading_cost(symbol_flow[0], symbol_flow[1]) + voice_leading_cost(symbol_flow[0], symbol_flow[2])
#         elif polytope_pattern == [[1,(1,1)]]:
#             f = mvt.triadic_mvt_chords(symbol_flow[0], symbol_flow[1])            
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[1])
#             fictive_element = mvt.apply_triadic_mvt(symbol_flow[1], f)
#             cost += voice_leading_cost(fictive_element, symbol_flow[2])
#             return cost
#         else:
#             raise err.PatternToDebugError("Uknonwn pattern: " + str(polytope_pattern))

#     if pattern_size == 4:
#         if polytope_pattern == [[1,1],[1,1]]:
#             return s_and_c_cost(symbol_flow)
#         elif polytope_pattern == [[1,(1,1)],[1]]:
#             f = mvt.triadic_mvt_chords(symbol_flow[0], symbol_flow[1])            
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[1])
#             fictive_element = mvt.apply_triadic_mvt(symbol_flow[1], f)
#             cost += voice_leading_cost(fictive_element, symbol_flow[2])
#             cost += voice_leading_cost(symbol_flow[0], symbol_flow[3])
#             return cost
#         elif polytope_pattern == [[1,1],[(1,1)]]:
#             g = mvt.triadic_mvt_chords(symbol_flow[0], symbol_flow[2])
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[1])
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[2])
#             fictive_element = mvt.apply_triadic_mvt(symbol_flow[2], g)
#             cost += voice_leading_cost(fictive_element, symbol_flow[3])
#             return cost
#         else:
#             raise err.PatternToDebugError("Uknonwn pattern: " + str(polytope_pattern))
#     if pattern_size == 5:
#         if polytope_pattern == [[1,1],[1,(1,1)]]:
#             cost = s_and_c_cost(symbol_flow[:4])
#             cost += voice_leading_cost(symbol_flow[3], symbol_flow[4])
#             return cost
#         else:
#             raise err.PatternToDebugError("Uknonwn pattern: " + str(polytope_pattern))
#     if pattern_size == 6:
#         if polytope_pattern == [[1,1],[(1,1),(1,1)]]:
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[1])
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[2])
#             f = mvt.triadic_mvt_chords(symbol_flow[0], symbol_flow[2])
#             first_fictive_element = mvt.apply_triadic_mvt(symbol_flow[2], f)
#             cost += voice_leading_cost(first_fictive_element, symbol_flow[3])
            
#             snd_fictive_element = mvt.apply_triadic_mvt(symbol_flow[1], f)
#             cost += voice_leading_cost(snd_fictive_element, symbol_flow[4])
            
#             trd_fictive_element = mvt.apply_triadic_mvt(symbol_flow[4], f)
#             cost += voice_leading_cost(trd_fictive_element, symbol_flow[5])
#             return cost

#         elif polytope_pattern == [[1,(1,1)],[1,(1,1)]]:
#             f = mvt.triadic_mvt_chords(symbol_flow[0], symbol_flow[1])
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[1])
#             first_fictive_element = mvt.apply_triadic_mvt(symbol_flow[1], f)
#             cost += voice_leading_cost(first_fictive_element, symbol_flow[2])
            
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[3])
#             snd_fictive_element = mvt.apply_triadic_mvt(symbol_flow[3], f)
#             cost += voice_leading_cost(snd_fictive_element, symbol_flow[4])
            
#             trd_fictive_element = mvt.apply_triadic_mvt(symbol_flow[4], f)
#             cost += voice_leading_cost(trd_fictive_element, symbol_flow[5])
#             return cost
        
#         elif polytope_pattern == [[(1,1),(1,1)],[(1,1)]]:
#             f = mvt.triadic_mvt_chords(symbol_flow[0], symbol_flow[1])
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[1])
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[2])
#             first_fictive_element = mvt.apply_triadic_mvt(symbol_flow[2], f)
#             cost += voice_leading_cost(first_fictive_element, symbol_flow[3])
            
#             cost = voice_leading_cost(symbol_flow[0], symbol_flow[4])
#             snd_fictive_element = mvt.apply_triadic_mvt(symbol_flow[4], f)
#             cost += voice_leading_cost(snd_fictive_element, symbol_flow[5])
#             return cost
#         else:
#             raise err.PatternToDebugError("Uknonwn pattern: " + str(polytope_pattern))
#     if pattern_size == 8:
#         return polytopic_scale_s_and_c_cost_computation(symbol_flow, pf.make_regular_polytope_pattern(3))
#     else:
#         raise err.PatternToDebugError("Uknonwn pattern size: " + str(polytope_pattern))

# %% System and contrast definition
# Triadic optimization of a System and Contrast
def s_and_c_cost(four_chords, relation_type = "triad_circle"):
    """
    Compute the cost of these four chords in the System and Cotnrast paradigm, with the cost function for relation defined in 'measure'.
    
    The 'chromatic' argument is there to differentiate between two different systems of relation (triadic circles), but in fact only one is used, the circle of fifth.

    Parameters
    ----------
    four_chords : Chords, in any form
        The four chords to evaluate in S&C.
    measure : function name, optional
        The type of cost function for relation. The default is mvt.l1_norm (i.e. the absolute value).
    chromatic: boolean
        If True, the chords in the circle of triads are ordered in the chromatic order,
        If False, the chords in the circle of triads are ordered in the 3-5 Torus order.
        Default: True (in fact False is not defined anymore).
        
    Returns
    -------
    cost : integer 
        the cost of this S&C.
    """
    cost = score_relation_switcher(relation_type, four_chords[0], four_chords[1])
    cost += score_relation_switcher(relation_type,four_chords[0], four_chords[2])
    rel = find_relation_switcher(relation_type,four_chords[0], four_chords[1])
    fictive_element = apply_relation_switcher(relation_type,four_chords[2], rel)
    cost += score_relation_switcher(relation_type,fictive_element, four_chords[3])
    return cost

# Cost function between two chords
# def voice_leading_cost(first_chord, second_chord, measure = mvt.l1_norm, triadic = False, fifth = False, chromatic = True):
#     """
#     Compute the score/cost associated with a voice leading movement between two chords.
    
#     Parameters
#     ----------
#     first_chord, second_chord: Chord objects
#         The chords between which the voice leading is to compute.
#     measure: function of mvt
#         The norm of the relation vector, defining the distance
#         (implemented: l1, l2 and infinite norm).
#         Default: l1_norm
#     triadic: boolean
#         If True, the movement between the chords is computed as a rotation in the circle of triads.
#         If False, the movement is computed in the optimal transport paradigm.
#         Default: False
#     fifth: boolean
#         Only useful if triadic is set to False.
#         If True, the transport between 2 notes is computed as a movement in the circle of fifth,
#         If False, the transport is the difference of the numbers of the notes (second - first).
#         # NB: Optimal transport is not used anymore
#         Default: False.
#     chromatic: boolean
#         Only useful if triadic is set to True.
#         If True, the chords in the circle of triads are ordered in the chromatic order,
#         If False, the chords in the circle of triads are ordered in the 3-5 Torus order.
#         Default: True
        
#     Returns
#     -------
#     integer: 
#         the cost of the voice leading.
            
#     """
#     return measure(mvt.triadic_mvt_chords(first_chord, second_chord, chromatic = chromatic))


def best_louboutin_cost_segment(segment, irregularity_penalty = 0, target_size = 32, segment_size_penalty = 0, relation_type = "triad_circle"):
    """
    Compute the optimal cost in the C. Guichaoua's paradigm, for this chord_sequence, among all possible patterns.
    
    Disclaimer: This function should be used for tests/demonstration ONLY, as it computes patterns and antecedents/successors when called.
    Prefer guichaoua_cost() for optimization at a music piece scale.
    
    Parameters
    ----------
    segment : list of Chords, in any form
        The segment, on which to compute the cost.
    positive_penalty, negative_penalty : float/integer, optional
        Penalty parameter related to irregularities of the polytope.
        Positive corresponds to the penalty when the polytope contains addition, negative is for deletion.
        They are constants and not function of the size of irregularities.
    positive_segment_size_penalty, negative_segment_size_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the size of the segment.
        positive_segment_size_penalty is the parameter when size exceeds 'target_size', negative is for size shorter than 'target_size'.
    target_size : integer, optional
        The optimal size, used for the penalty related to the segment size. 
        The default is 32.

    Returns
    -------
    this_segment_cost : integer
        Optimal cost for this segment in the C. Guichaoua's paradigm.
    best_pattern : nested list of integers (indexed pattern)
        The pattern, resulting in the optimal score.
    
    """
    this_bag = sh.compute_patterns_and_ppp_for_size(len(segment))
    this_segment_cost = math.inf

    if this_bag == []:
        return this_segment_cost
                
    for a_pattern in this_bag:
        this_polytope_cost = math.inf
        for i in range(len(a_pattern[0])):
            this_ppp_cost = louboutin_cost_for_a_ppp(segment, a_pattern[0][i], a_pattern[3][i], a_pattern[4][i], current_min = this_segment_cost, relation_type = relation_type)
            if this_ppp_cost < this_polytope_cost:
                this_polytope_cost = this_ppp_cost
                best_ppp = a_pattern[0][i]
        
        this_polytope_cost += irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = irregularity_penalty, negative_penalty = irregularity_penalty)

        if this_polytope_cost < this_segment_cost:
            this_segment_cost = this_polytope_cost
            best_pattern = best_ppp
            
    this_segment_cost += sh.penalty_cost_guichaoua(len(segment), target_size = target_size, positive_segment_size_penalty = segment_size_penalty, negative_segment_size_penalty = segment_size_penalty)
    return this_segment_cost, best_pattern

# %% Guichaoua paradigm
def guichaoua_cost(chord_sequence, indexed_pattern, antecedents_with_pivots, successors, correct_antecedents, current_min = math.inf, relation_type = "triad_circle"):
    """
    Compute the cost in the C. Guichaoua's paradigm, for this chord_sequence and this 'indexed_pattern'.
    
    Antecedents, pivots and successors are given as parameters so as they don't need to be recomputed at each iteration.

    Parameters
    ----------
    chord_sequence : list of Chords, in any form
        The chord sequence.
    indexed_pattern : nested list of integers
        The indexed pattern, to compute score on.
    antecedents_with_pivots : list of list of tuples (integer, integer)
        Antedents with their pivots, for each element.
        Given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.
    successors : list of list of integers
        List of successors, for each element, given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.
    correct_antecedents : list of list of integers
        Initial list of valid antecedents, for each element, given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.

    Raises
    ------
    PatternToDebugError
        Error raised when an element has no antecedent (which is not normal behavior).

    Returns
    -------
    score : integer
        Cost of this sequence on this pattern in the C. Guichaoua's paradigm.

    """
    score = 1
    for elt in range(1, pf.get_pattern_size(indexed_pattern)):
        if score > current_min:
            return math.inf
        this_elt_ant = antecedents_with_pivots[elt]
        if this_elt_ant == []:
            raise err.PatternToDebugError("Element with no antecedent: {} in {}? This shouldn't happen a priori.".format(elt, indexed_pattern))
        elif (0,0) in this_elt_ant:
            if chord_sequence[0] != chord_sequence[elt]:
                score += 1
        else:
            # If this element doesn't hold valid predecessors.
            if correct_antecedents[elt] == []:
                score += 1

                # Update the correct antecedents of this element' successors
                for this_elt_successor in successors[elt]:
                    correct_antecedents[this_elt_successor] = sh.update_correct_antecedents(elt, antecedents_with_pivots[this_elt_successor], correct_antecedents[this_elt_successor])
            # Among the valid predecessors, search for a valid implication
            else:
                # Searching for a valid implication of the current element.
                found = False
                for ant, piv in this_elt_ant:
                    if ant in correct_antecedents[elt]:
                        rel = find_relation_switcher(relation_type, chord_sequence[0], chord_sequence[ant])
                        if rel == find_relation_switcher(relation_type, chord_sequence[piv], chord_sequence[elt]):
                            found = True
                if not found:
                    score += 1

                    # Update the correct antecedents of this element' successors
                    for this_elt_successor in successors[elt]:
                        correct_antecedents[this_elt_successor] = sh.update_correct_antecedents(elt, antecedents_with_pivots[this_elt_successor], correct_antecedents[this_elt_successor])

    return score

def guichaoua_cost_global_antecedents_successors(chord_sequence, indexed_pattern, antecedents_with_pivots, successors, correct_antecedents, current_min = math.inf, relation_type = "triad_circle"):
    """
    Compute the cost in the C. Guichaoua's paradigm, but with antecedents as "global" and not just the ones linked to the element by a direct arrow (original behavior from my point of view).
    
    Antecedents, pivots and successors are given as parameters so as they don't need to be recomputed at each iteration.

    Parameters
    ----------
    chord_sequence : list of Chords, in any form
        The chord sequence.
    indexed_pattern : nested list of integers
        The indexed pattern, to compute score on.
    antecedents_with_pivots : list of list of tuples (integer, integer)
        Antedents with their pivots, for each element. Here, antecedents are global ones, so 0 is antecedent to every element.
        Given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.
    successors : list of list of integers
        List of successors, for each element. Successors are global too.
        Given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.
    correct_antecedents : list of list of integers
        Initial list of valid antecedents, for each element, given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.

    Raises
    ------
    PatternToDebugError
        Error raised when an element has no antecedent (which is not normal behavior).

    Returns
    -------
    score : integer
        Cost of this sequence on this pattern in the C. Guichaoua's paradigm, with global antecedents and successors.

    """
    score = 1
    for elt in range(1, pf.get_pattern_size(indexed_pattern)):
        if score > current_min:
            return math.inf
        this_elt_ant = antecedents_with_pivots[elt]
        if this_elt_ant == []:
            raise err.PatternToDebugError("Element with no antecedent: {} in {}. This shouldn't happen a priori.".format(elt, indexed_pattern))
        elif (0,0) in this_elt_ant:
            if chord_sequence[0] != chord_sequence[elt]:
                score += 1
        else:
            # If this element doesn't hold valid predecessors.
            if correct_antecedents[elt] == []:
                score += 1

                # Update the correct antecedents of this element' successors
                for this_elt_successor in successors[elt]:
                    this_elt_successor_ant_piv = antecedents_with_pivots[this_elt_successor]
                    pivot = None
                    intersected_successors_and_all = [elt]
                    for couple in this_elt_successor_ant_piv:
                        if couple[0] == elt:
                            pivot = couple[1]
                            intersected_successors_and_all.append(pivot)
                        elif couple[1] == elt:
                            pivot = couple[0]
                            intersected_successors_and_all.append(pivot)
                        elif couple[0] in successors[elt]:
                            intersected_successors_and_all.append(couple[0])
                            intersected_successors_and_all.append(couple[1])
                        elif couple[1] in successors[elt]:
                            intersected_successors_and_all.append(couple[0])
                            intersected_successors_and_all.append(couple[1])
                    if pivot == None:
                        raise err.ToDebugException("Pivot shouldn't be null: Element: {}, (ant,piv): {} for successor {}, indexed_pattern: {}.".format(elt, this_elt_successor_ant_piv, this_elt_successor, indexed_pattern))
                    
                    new_correct_antecedents = []
                    for ant_to_update in correct_antecedents[this_elt_successor]:
                        if ant_to_update in intersected_successors_and_all:
                            new_correct_antecedents.append(ant_to_update)
                    correct_antecedents[this_elt_successor] = new_correct_antecedents
                # Among the valid predecessors, search for a valid implication
            else:
                # Searching for a valid implication of the current element.
                found = False
                for ant, piv in this_elt_ant:
                    if ant in correct_antecedents[elt]:
                        rel = find_relation_switcher(relation_type,chord_sequence[0], chord_sequence[ant])
                        if rel == find_relation_switcher(relation_type,chord_sequence[piv], chord_sequence[elt]):
                            found = True
                if not found:
                    score += 1

                    # Update the correct antecedents of this element' successors
                    for this_elt_successor in successors[elt]:
                        this_elt_successor_ant_piv = antecedents_with_pivots[this_elt_successor]
                        pivot = None
                        intersected_successors_and_all = [elt]
                        for couple in this_elt_successor_ant_piv:
                            if couple[0] == elt:
                                pivot = couple[1]
                                intersected_successors_and_all.append(pivot)
                            elif couple[1] == elt:
                                pivot = couple[0]
                                intersected_successors_and_all.append(pivot)
                            elif couple[0] in successors[elt]:
                                intersected_successors_and_all.append(couple[0])
                                intersected_successors_and_all.append(couple[1])
                            elif couple[1] in successors[elt]:
                                intersected_successors_and_all.append(couple[0])
                                intersected_successors_and_all.append(couple[1])
                        if pivot == None:
                            raise err.ToDebugException("Pivot shouldn't be null: Element: {}, suc:{}, (ant,piv): {} for successor {}, indexed_pattern: {}.".format(elt, successors[elt], this_elt_successor_ant_piv, this_elt_successor, indexed_pattern))
                        
                        new_correct_antecedents = []
                        for ant_to_update in correct_antecedents[this_elt_successor]:
                            if ant_to_update in intersected_successors_and_all:
                                new_correct_antecedents.append(ant_to_update)
                        correct_antecedents[this_elt_successor] = new_correct_antecedents
    return score

def best_guichaoua_cost_segment(segment, positive_penalty = 0, negative_penalty = 0, target_size = 32, 
                                positive_segment_size_penalty = 0, negative_segment_size_penalty = 0):
    """
    Compute the optimal cost in the C. Guichaoua's paradigm, for this chord_sequence, among all possible patterns.
    
    Disclaimer: This function should be used for tests/demonstration ONLY, as it computes patterns and antecedents/successors when called.
    Prefer guichaoua_cost() for optimization at a music piece scale.
    
    Parameters
    ----------
    segment : list of Chords, in any form
        The segment, on which to compute the cost.
    positive_penalty, negative_penalty : float/integer, optional
        Penalty parameter related to irregularities of the polytope.
        Positive corresponds to the penalty when the polytope contains addition, negative is for deletion.
        They are constants and not function of the size of irregularities.
    positive_segment_size_penalty, negative_segment_size_penalty : float/integer, optional
        Penalty parameter to multiply to the raw penalty score for the size of the segment.
        positive_segment_size_penalty is the parameter when size exceeds 'target_size', negative is for size shorter than 'target_size'.
    target_size : integer, optional
        The optimal size, used for the penalty related to the segment size. 
        The default is 32.

    Returns
    -------
    this_segment_cost : integer
        Optimal cost for this segment in the C. Guichaoua's paradigm.
    best_pattern : nested list of integers (indexed pattern)
        The pattern, resulting in the optimal score.
    
    """
    this_bag = sh.compute_patterns_with_antecedents_for_size(len(segment))
    this_segment_cost = math.inf

    if this_bag == []:
        print("No polytope for this size: {}".format(len(segment)))
        return this_segment_cost
                
    for a_pattern in this_bag:
        this_polytope_cost = guichaoua_cost(segment, a_pattern[0], a_pattern[3], a_pattern[4], a_pattern[5], current_min = this_segment_cost)
            
        this_polytope_cost += irregularities_penalty_guichaoua(adding_code = a_pattern[1], deleting_code = a_pattern[2], positive_penalty = positive_penalty, negative_penalty = negative_penalty)
        if this_polytope_cost < this_segment_cost:
            this_segment_cost = this_polytope_cost
            best_pattern = a_pattern[0]
            
    this_segment_cost += sh.penalty_cost_guichaoua(len(segment), target_size = target_size, positive_segment_size_penalty = positive_segment_size_penalty, negative_segment_size_penalty = negative_segment_size_penalty)
    return this_segment_cost, best_pattern

# %% New paradigms, developed by A. Marmoret
def louboutaoua_cost(chord_sequence, indexed_pattern, antecedents_with_pivots, successors, correct_antecedents, direct_antecedents, current_min = math.inf, relation_type = "triad_circle"):
    """
    Louboutaoua cost (name still pending). Need a reference to be explained.

    Parameters
    ----------
    chord_sequence : list of Chords, in any form
        The chord sequence.
    indexed_pattern : nested list of integers
        The indexed pattern, to compute score on.
    antecedents_with_pivots : list of list of tuples (integer, integer)
        Antedents with their pivots, for each element.
        Given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.
    successors : list of list of integers
        List of successors, for each element, given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.
    correct_antecedents : list of list of integers
        Initial list of valid antecedents, for each element, given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.
    direct_antecedents : list of integers or tuples of three integers
        The direct antecedent in C. Louboutin's paradigm (only one antecedent for each element).
    current_min : integer, optional
        The current minimal cost.
        It's an acceleration parameter, to check that the current cost we're computing' isn't already higher than the current minimal value.
        If it's the case, this ppp won't be chosen, so we don't need an exact score, and it returns math.inf.
        The default is math.inf.

    Raises
    ------
    PatternToDebugError
        Error raised when an element has no antecedent (which is not normal behavior).

    Returns
    -------
    score : integer
        Cost of this sequence on this pattern (and this ppp) in the Louboutaoua's paradigm.

    """
    score = 1
    for elt in range(1, pf.get_pattern_size(indexed_pattern)):
        if score > current_min:
            return math.inf
        this_elt_ant = antecedents_with_pivots[elt]
        if this_elt_ant == []:
            raise err.PatternToDebugError("Element with no antecedent: {} in {}. This shouldn't happen a priori.".format(elt, indexed_pattern))
        elif (0,0) in this_elt_ant:
            if chord_sequence[0] != chord_sequence[elt]:
                score += score_relation_switcher(relation_type, chord_sequence[0], chord_sequence[elt])
        else:
            # If this element doesn't hold valid predecessors.
            if correct_antecedents[elt] == []:
                direct_ant = direct_antecedents[elt]
                if type(direct_ant) is tuple: # Antecedent is a fictive element, to construct
                    f = find_relation_switcher(relation_type,chord_sequence[direct_ant[0]], chord_sequence[direct_ant[1]])            
                    fictive_element = apply_relation_switcher(relation_type,chord_sequence[direct_ant[2]], f)
                    score += score_relation_switcher(relation_type, fictive_element, chord_sequence[elt])
                else:
                    score += score_relation_switcher(relation_type,chord_sequence[direct_ant], chord_sequence[elt])

                # Update the correct antecedents of this element' successors
                for this_elt_successor in successors[elt]:
                    correct_antecedents[this_elt_successor] = sh.update_correct_antecedents(elt, antecedents_with_pivots[this_elt_successor], correct_antecedents[this_elt_successor])
            # Among the valid predecessors, search for a valid implication
            else:
                # Searching for a valid implication of the current element.
                found = False
                contrasts = []
                for ant, piv in this_elt_ant:
                    if ant in correct_antecedents[elt]:
                        rel = find_relation_switcher(relation_type,chord_sequence[0], chord_sequence[ant])
                        fictive = apply_relation_switcher(relation_type,chord_sequence[piv], rel)
                        gamma = find_relation_switcher(relation_type,fictive, chord_sequence[elt])
                        if gamma == 0:
                            found = True
                        else:
                            contrasts.append(gamma)
                if not found:
                    if len(np.unique(contrasts)) == 1:
                        score += abs(contrasts[0])
                    else:
                        direct_ant = direct_antecedents[elt]
                        if type(direct_ant) is tuple: # Antecedent is a fictive element, to construct
                            f = find_relation_switcher(relation_type,chord_sequence[direct_ant[0]], chord_sequence[direct_ant[1]])            
                            fictive_element = apply_relation_switcher(relation_type,chord_sequence[direct_ant[2]], f)
                            score += score_relation_switcher(relation_type,fictive_element, chord_sequence[elt])
                        else:
                            score += score_relation_switcher(relation_type,chord_sequence[direct_ant], chord_sequence[elt])
                    # Update the correct antecedents of this element' successors
                    for this_elt_successor in successors[elt]:
                        correct_antecedents[this_elt_successor] = sh.update_correct_antecedents(elt, antecedents_with_pivots[this_elt_successor], correct_antecedents[this_elt_successor])

    return score

# def cohen_marmoret_cost(chord_sequence, indexed_pattern, antecedents_with_pivots, current_min = math.inf, relation_type = "triad_circle"):
#     """
#     Cohen-Marmoret cost (name still pending). Need a reference to be explained.

#     Parameters
#     ----------
#     chord_sequence : list of Chords, in any form
#         The chord sequence.
#     indexed_pattern : nested list of integers
#         The indexed pattern, to compute score on.
#     antecedents_with_pivots : list of list of tuples (integer, integer)
#         Antedents with their pivots, for each element.
#         Given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
#         It's an acceleration technique.
#     current_min : integer, optional
#         The current minimal cost.
#         It's an acceleration parameter, to check that the current cost we're computing' isn't already higher than the current minimal value.
#         If it's the case, this ppp won't be chosen, so we don't need an exact score, and it returns math.inf.
#         The default is math.inf.

#     Raises
#     ------
#     PatternToDebugError
#         Error raised when an element has no antecedent (which is not normal behavior).

#     Returns
#     -------
#     score : integer
#         Cost of this sequence on this pattern in the Cohen-Marmoret's paradigm.

#     """
#     print("Should pass through this cost function again, because contrast are all scores now (and not functions). See if it breaks sthg")
#     score = 1
#     contrasts = [None for i in range(pf.get_pattern_size(indexed_pattern))]
#     for elt in range(1, pf.get_pattern_size(indexed_pattern)):
#         if score > current_min:
#             return math.inf
#         this_elt_ant = antecedents_with_pivots[elt]
#         if len(this_elt_ant) == 2 and (this_elt_ant[0][0], this_elt_ant[0][1]) == (this_elt_ant[1][1], this_elt_ant[1][0]):
#             this_elt_ant = [this_elt_ant[0]]
#         if this_elt_ant == []:
#             raise err.PatternToDebugError("Element with no antecedent: {} in {}. This shouldn't happen a priori.".format(elt, indexed_pattern))
#         elif (0,0) in this_elt_ant:
#             if chord_sequence[0] != chord_sequence[elt]:
#                 score += score_relation_switcher(relation_type, chord_sequence[0], chord_sequence[elt])
#         else:
#             # Searching for a valid implication of the current element.
#             if len(this_elt_ant) == 1:
#                 ant, piv = this_elt_ant[0]
#                 rel = find_relation_switcher(relation_type,chord_sequence[0], chord_sequence[ant])
#                 fictive = apply_relation_switcher(relation_type,chord_sequence[piv], rel)
#                 #contrast = find_relation_switcher(relation_type, fictive, chord_sequence[elt])
#                 contrast_score = score_relation_switcher(relation_type,fictive, chord_sequence[elt])

#             elif len(this_elt_ant) == 0:
#                 raise err.PatternToDebugError("Element with no antecedent: {}, antecedents and pivots for it: {} in {}. This shouldn't happen a priori.".format(elt, this_elt_ant, indexed_pattern))
#             else:
#                 contrasts_of_ants = []
#                 possible_fictive = []
#                 for ant, piv in this_elt_ant:
#                     if contrasts[ant] == None:
#                         raise err.PatternToDebugError("Contrast for the element: {}, hasn't been computed when looping for element: {} in {}. This shouldn't happen a priori.".format(ant, elt, indexed_pattern))
#                     else:
#                         contrasts_of_ants.append(contrasts[ant])
#                         rel = find_relation_switcher(relation_type, chord_sequence[0], chord_sequence[ant])
#                         fictive = apply_relation_switcher(relation_type, chord_sequence[piv], rel)
#                         possible_fictive.append(fictive)

#                 if len(np.unique(possible_fictive)) == 1: # No ambiguity
#                     #contrast = find_relation_switcher(relation_type,possible_fictive[0], chord_sequence[elt])
#                     contrast_score = score_relation_switcher(relation_type,possible_fictive[0], chord_sequence[elt])
#                 else:
#                     the_max = -1
#                     maximal_contrastic_ants = None
#                     for idx in range(len(contrasts_of_ants)): # Looping among the antecedents of our current element, and searching for the maximally contrastic one. If they are several, we will evaluate them all.
#                         absolute_val_contrast = contrasts_of_ants[idx] # Adding a score now, so it will always be positive
#                         if absolute_val_contrast > the_max: # Higher contrast than the previous ones.
#                             the_max = absolute_val_contrast
#                             maximal_contrastic_ants = [idx]
#                         elif absolute_val_contrast == the_max: # Contrast equal to the max, to evaluate.
#                             maximal_contrastic_ants.append(idx)
                            
#                     min_contrast_score_among_valids = math.inf
#                     for ant in maximal_contrastic_ants: # Happens that all contrasts are equal, but that they result in different fictive. In that case, we keep the minimal contrast.
#                         #current_contrast = find_relation_switcher(relation_type, possible_fictive[ant], chord_sequence[elt])
#                         current_contrast_score = score_relation_switcher(relation_type, possible_fictive[ant], chord_sequence[elt])
#                         if current_contrast_score < min_contrast_score_among_valids:
#                             min_contrast_score_among_valids = current_contrast_score
#                             #min_contrast_among_valids = current_contrast
#                     if min_contrast_score_among_valids == math.inf:
#                         raise err.ToDebugException("Infinite contrast.")
#                     contrast_score = min_contrast_score_among_valids
                    
#             contrasts[elt] = contrast_score
#             score += contrast_score

#     return score
    
def cohen_marmoret_cost(chord_sequence, indexed_pattern, antecedents_with_pivots, current_min = math.inf, relation_type = "triad_circle"):
    """
    Cohen-Marmoret cost (name still pending). Need a reference to be explained.

    Parameters
    ----------
    chord_sequence : list of Chords, in any form
        The chord sequence.
    indexed_pattern : nested list of integers
        The indexed pattern, to compute score on.
    antecedents_with_pivots : list of list of tuples (integer, integer)
        Antedents with their pivots, for each element.
        Given as arguments so as they can be computed once and then feeded when needed in the function (when segmenting a song).
        It's an acceleration technique.
    current_min : integer, optional
        The current minimal cost.
        It's an acceleration parameter, to check that the current cost we're computing' isn't already higher than the current minimal value.
        If it's the case, this ppp won't be chosen, so we don't need an exact score, and it returns math.inf.
        The default is math.inf.

    Raises
    ------
    PatternToDebugError
        Error raised when an element has no antecedent (which is not normal behavior).

    Returns
    -------
    score : integer
        Cost of this sequence on this pattern in the Cohen-Marmoret's paradigm.

    """
    score = 1
    contrasts = [None for i in range(pf.get_pattern_size(indexed_pattern))]
    for elt in range(1, pf.get_pattern_size(indexed_pattern)):
        if score > current_min:
            return math.inf
        this_elt_ant = antecedents_with_pivots[elt]
        if len(this_elt_ant) == 2 and (this_elt_ant[0][0], this_elt_ant[0][1]) == (this_elt_ant[1][1], this_elt_ant[1][0]):
            this_elt_ant = [this_elt_ant[0]]
        if this_elt_ant == []:
            raise err.PatternToDebugError("Element with no antecedent: {} in {}. This shouldn't happen a priori.".format(elt, indexed_pattern))
        elif (0,0) in this_elt_ant:
            #print("la")
            if chord_sequence[0] != chord_sequence[elt]:
                score += score_relation_switcher(relation_type, chord_sequence[0], chord_sequence[elt])
        else:
            #print("lab")
            # Searching for a valid implication of the current element.
            if len(this_elt_ant) == 1:
                #print("1ant")
                ant, piv = this_elt_ant[0]
                rel = find_relation_switcher(relation_type, chord_sequence[0], chord_sequence[ant])
                fictive = apply_relation_switcher(relation_type, chord_sequence[piv], rel)
                contrast = find_relation_switcher(relation_type, fictive, chord_sequence[elt])

            elif len(this_elt_ant) == 0:
                raise err.PatternToDebugError("Element with no antecedent: {}, antecedents and pivots for it: {} in {}. This shouldn't happen a priori.".format(elt, this_elt_ant, indexed_pattern))
            else:
                #print("Several ant")

                contrasts_of_ants = []
                possible_fictive = []
                for ant, piv in this_elt_ant:
                    if contrasts[ant] == None:
                        raise err.PatternToDebugError("Contrast for the element: {}, hasn't been computed when looping for element: {} in {}. This shouldn't happen a priori.".format(ant, elt, indexed_pattern))
                    else:
                        contrasts_of_ants.append(contrasts[ant])
                        rel = find_relation_switcher(relation_type, chord_sequence[0], chord_sequence[ant])
                        fictive = apply_relation_switcher(relation_type, chord_sequence[piv], rel)
                        possible_fictive.append(fictive)

                if len(np.unique(possible_fictive)) == 1: # No ambiguity
                    #print("No amb")

                    contrast = find_relation_switcher(relation_type, possible_fictive[0], chord_sequence[elt])
                else:
                    the_max = -1
                    maximal_contrastic_ants = None
                    for idx in range(len(contrasts_of_ants)): # Looping among the antecedents of our current element, and searching for the maximally contrastic one. If they are several, we will evaluate them all.
                        #print("La la ouais")

                        score_contrast = score_one_relation_switcher(relation_type, contrasts_of_ants[idx])
                        if score_contrast > the_max: # Higher contrast than the previous ones.
                            the_max = score_contrast
                            maximal_contrastic_ants = [idx]
                        elif score_contrast == the_max: # Contrast equal to the max, to evaluate.
                            maximal_contrastic_ants.append(idx)
                            
                    min_contrast_among_valids = math.inf # Careful: if the norm changes, this bounds has too!!!
                    for ant in maximal_contrastic_ants: # Happens that all contrasts are equal, but that they result in different fictive. In that case, we keep the minimal contrast.
                        #print("Raise marche pas")

                        current_contrast = find_relation_switcher(relation_type, possible_fictive[ant], chord_sequence[elt])
                        if score_one_relation_switcher(relation_type, current_contrast) < score_one_relation_switcher(relation_type, min_contrast_among_valids):
                            min_contrast_among_valids = current_contrast
                    if min_contrast_among_valids == math.inf:
                        raise NotImplementedError("Infinite contrast")
                    contrast = min_contrast_among_valids
                    
            contrasts[elt] = contrast
            score += score_one_relation_switcher(relation_type, contrast)

    return score


# %% Relation system
def find_relation_switcher(relation_type, chord_1, chord_2):
    if relation_type == "triad_circle":
        return tt.triadic_mvt_triads(chord_1, chord_2)
    elif relation_type == "chromatic_circle":
        return acc_pc.accelerated_chromatic_mvt_triads(chord_1, chord_2)
    elif relation_type == "3_5_torus":
        return tt.three_five_torus_mvt_triads(chord_1, chord_2)
    elif relation_type == "tonnetz":
        return acc_pc.accelerated_triadic_tonnetz_relation_symbol(chord_1, chord_2)
    elif relation_type == "voice_leading":
        return acc_pc.accelerated_get_voice_leading_transformation_symbol(chord_1, chord_2)
    else:
        raise err.InvalidArgumentValueException(f"Invalid relation_type: {relation_type}")
    
def apply_relation_switcher(relation_type, chord_1, relation):
    if relation_type == "triad_circle":
        return tt.apply_triadic_mvt(chord_1, relation)
    elif relation_type == "tonnetz":
        return acc_pc.accelerated_apply_triadic_tonnetz_relation_symbol(chord_1, relation)
    elif relation_type == "voice_leading":
        raise NotImplementedError("TODO")
    else:
        raise err.InvalidArgumentValueException(f"Invalid relation_type: {relation_type}")
        
def score_relation_switcher(relation_type, chord_1, chord_2):
    if relation_type == "triad_circle":
        return abs(tt.triadic_mvt_triads(chord_1, chord_2))
    elif relation_type == "chromatic_circle":
        return abs(tt.chromatic_mvt_triads(chord_1, chord_2))
    elif relation_type == "3_5_torus":
        return abs(tt.three_five_torus_mvt_triads(chord_1, chord_2))
    elif relation_type == "tonnetz":
        return acc_pc.accelerated_triadic_tonnetz_distance_symbol(chord_1, chord_2)
    elif relation_type == "voice_leading":
        return tt.get_voice_leading_distance_symbol(chord_1, chord_2)
        raise NotImplementedError("TODO")
    else:
        raise err.InvalidArgumentValueException(f"Invalid relation_type: {relation_type}")
        
def score_one_relation_switcher(relation_type, rel):
    if rel == math.inf:
        return math.inf
    if relation_type == "triad_circle":
        return abs(rel)
    elif relation_type == "tonnetz":
        if rel == 0:
            return 0
        return len(rel)
    elif relation_type == "voice_leading":
        raise NotImplementedError("TODO")
    else:
        raise err.InvalidArgumentValueException(f"Invalid relation_type: {relation_type}")

# %% Sequential score
def sequential_score(chord_flow, penalty, relation_type = "triad_circle"):
    """
    Sequential score for this sequence (score where relations are taken in the chronological order, useful as a baseline).

    Parameters
    ----------
    chord_flow : list of Chords
        The chord sequence.
    penalty : integer
        The penalty cost (is it necessary?).

    Returns
    -------
    integer
        Sequential score of this sequence.

    """
    score = 0
    for two_chords in zip(chord_flow[:-1], chord_flow[1:]):
        score += score_relation_switcher(relation_type, two_chords[0], two_chords[1])
    return score + penalty

# %% Penalties form irregularities in polytopes
def irregularities_penalty(adding_code, deleting_code, penalty = 20):
    """
    Penalty, to add to the score, which is proportional to the total number of alteration in the polytope.

    Parameters
    ----------
    adding_code : list of binary numbers
        The adding code for generating the pattern.
    deleting_code : list of binary numbers
        The deleting code for generating the pattern.
    penalty : float/integer, optional
        The parameter which will be multiplied to the number of alteration. The default is 20.

    Raises
    ------
    WrongIrregularCode
        Raises an error if both irregular codes are of different lengths.

    Returns
    -------
    integer/float
        The penalty score.

    """
    if adding_code == []:
        return penalty * pf.get_deformation_size(deleting_code)
    elif deleting_code == []:
        return penalty * pf.get_deformation_size(adding_code)
    else:
        # # Total size of deformation, even if overlap
        # return penalty * (pf.get_deformation_size(adding_code) + pf.get_deformation_size(deleting_code))
        
        # Neutralizing the overlap (Why should I do that? Because, in case of overlap, only suppression are taken in account)
        if len(adding_code) != len(deleting_code):
            raise err.WrongIrregularCode("Adding and deleting codes must be of same size") from None
        overlap_dimension = 0
        for index in range(len(adding_code)):
            if deleting_code[index] == 1 and adding_code[index] == 1:
                overlap_dimension += 1
        return penalty * (pf.get_deformation_size(adding_code) + pf.get_deformation_size(deleting_code) - 2**overlap_dimension)

def irregularities_penalty_guichaoua(adding_code, deleting_code, positive_penalty = 2.25, negative_penalty = 3):
    """
    Penalty, to add to the score, which is a function of the irregularities.
    
    It's the one used by C. Guichaoua in his work, and does not depend of the size of the irregularities,
    but is a constant added in function of the type of alteration.

    Parameters
    ----------
    adding_code : list of binary numbers
        The adding code for generating the pattern.
    deleting_code : list of binary numbers
        The deleting code for generating the pattern.
    positive_penalty : float/integer, optional
        The cost to add when there is addition on the polytope.
    negative_penalty : float/integer, optional
        The cost to add when there is deletion on the polytope.

    Returns
    -------
    integer/float
        The penalty score.

    """
    if adding_code == []:
        if deleting_code == []:
            return 0
        else:
            return negative_penalty
    elif deleting_code == []:
        return positive_penalty
    else:
        return positive_penalty + negative_penalty
    
# %% Specific functions, which shouldn't be used outside of the above functions.
def recursively_find_primers_chords(symbol_flow, pattern_of_ones):
    """
    Specific function for the 'polytopic_scale_s_and_c_cost_computation' function, in order to find the primers on a pattern of ones.
    
    For retrieving primers on an indexed pattern (which is a better behavior), use 'find_primers_of_low_level_systems' in pattern_manip.py

    Parameters
    ----------
    symbol_flow : list of Chords, in any forms
        The sequence of chords.
    pattern_of_ones : nested list of ones
        A pattern of ones.

    Raises
    ------
    PatternAndSequenceIncompatible
        Raises en error if the pattern and the sequence are of different sizes.

    Returns
    -------
    nested list of Chords
        The chords, as elements of a pattern.

    """
    # Recursively find all primers, to compute their cost.
    if pf.get_pattern_size(pattern_of_ones) != len(symbol_flow):
        raise err.PatternAndSequenceIncompatible("The pattern's length is different than the the chord sequence's length, which make them incompatible.") from None
    try:
        if pf.get_pattern_dimension(pattern_of_ones) <= 2:
            return symbol_flow[0]
        else:
            first_nested_pattern = pattern_of_ones[0]
            size_dim_inf = pf.get_pattern_size(first_nested_pattern)
            primer_1 = recursively_find_primers_chords(symbol_flow[:size_dim_inf], first_nested_pattern)
            if len(pattern_of_ones) == 2:
                primer_2 = recursively_find_primers_chords(symbol_flow[size_dim_inf:], pattern_of_ones[1])
            elif len(pattern_of_ones) != 1:
                raise err.PatternToDebugError("What is that pattern ? " + str(pattern_of_ones))
            else:
                return [recursively_find_primers_chords(symbol_flow, pattern_of_ones[0])]
            return [primer_1, primer_2]
    # except err.UnexpectedDim1Pattern:
    #     return None
    except IndexError: # Only one dim 2 pattern
        return recursively_find_primers_chords(symbol_flow, pattern_of_ones[0])
    
def extract_pattern_from_primers_chords(primers):
    """
    Specific function for the 'polytopic_scale_s_and_c_cost_computation' function, in order to extract a pattern of ones from the pattern of chords of function 'recursively_find_primers_chords'.
    
    To extract a pattern of one from an indexed pattern (which is a better behavior), use 'extract_pattern_from_indexed_pattern' in pattern_factory.py

    Parameters
    ----------
    primers : nested list of Chords
        The chords, as elements of a pattern (result of function 'recursively_find_primers_chords').

    Raises
    ------
    PatternToDebugError
        Raises en error if the pattern is incorrect.

    Returns
    -------
    nested list of one:
        Pattern of ones from the previous chord pattern.


    """
    # Extracting the pattern of the previously found primers (see find_all_primers).
    if not isinstance(primers, list):
        return 1
    elif len(primers) == 1:
        return [extract_pattern_from_primers_chords(primers[0])]
    elif len(primers) > 2:
        raise err.PatternToDebugError("Nest of more than 2 patterns.")
    else:
        return [extract_pattern_from_primers_chords(primers[0]), extract_pattern_from_primers_chords(primers[1])]


# %% Old functions
# # More arguments, good idea, to make the rest like that eventually
# # def triadic_optimization_of_4(four_chords, system = True, contrast = True, measure = mvt.l1_norm, chromatic = False):
# #     """
# #     Calculates the score of the four_chords with triadic relations and in the wanted paradigm (given by the system and contrast arguments).

# #     Parameters
# #     ----------
# #     four_chords: list of Chord objects
# #         The four chords which score is to compute.
# #     system, contrast: booleans
# #         If system is True, the score paradigm is will follow:
# #         f:0->1
# #         g:0->2
# #         whereas, if system is False, the chosen model will be sequential:
# #         f:0->1
# #         g:1->2
        
# #         If contrast is chosen, the third element will be computed as a contrastic element:
# #         f(g(0))->3
# #         If it is False, the paradigm will be
# #         h:0->3 (system = True)
# #         or
# #         h:2->3 (system = False)
# #         depending on the system parameter.
# #     measure: function of mvt
# #         The norm of the relation vector, defining the distance
# #         (implemented: l1, l2 and infinite norm).
# #         Default: l1_norm
# #     chromatic: boolean
# #         If True, the chords in the circle of triads are ordered in the chromatic order,
# #         If False, the chords in the circle of triads are ordered in the 3-5 Torus order.
# #         Default: True
# #     Returns
# #     -------
# #     four_chords : list of Chord objects
# #         The polytope's chords (for consistency between triadic and ots.
# #     cost : integer
# #         The score of this polytope in traidic relations.

# #     """
# #     cost = voice_leading_cost(four_chords[0], four_chords[1], triadic = True, measure = measure, chromatic = chromatic)
    
# #     if system:
# #         cost += voice_leading_cost(four_chords[0], four_chords[2], triadic = True, measure = measure, chromatic = chromatic)
# #     else:
# #         cost += voice_leading_cost(four_chords[1], four_chords[2], triadic = True, measure = measure, chromatic = chromatic)
        
# #     if contrast:
# #         rel = mvt.triadic_mvt_chords(four_chords[0], four_chords[1])
# #         fictive_element = mvt.apply_triadic_mvt(four_chords[2], rel)
# #         cost += voice_leading_cost(fictive_element, four_chords[3], triadic = True, measure = measure, chromatic = chromatic)
# #     elif system:
# #         cost += voice_leading_cost(four_chords[0], four_chords[3], triadic = True, measure = measure, chromatic = chromatic)
# #     else:
# #         cost += voice_leading_cost(four_chords[2], four_chords[3], triadic = True, measure = measure, chromatic = chromatic)
# #     return four_chords, cost
    
# On this one, patterns are precomputed (for acceleration purpose)
# def optimal_pattern_on_polytope_pre_computed_patterns(symbol_flow, patterns_this_large, penalty = 1, measure = mvt.l1_norm, chromatic = False, extended_s_and_c = False):
#     # TODO: handle measures and chromatic
#     if patterns_this_large == None or patterns_this_large == []:
#         #raise err.NoPolytopeForThisSize("No Polytope is available for this size of segment. Try sequential cost instead.") from None
#         warnings.warn("No Polytope is available for this size of segment. Trying sequential cost instead.")
#         return sequential_score(symbol_flow, 0), "Sequential"
#     min_cost = math.inf
#     best_pattern = None
#     for local_pattern, add, dele in patterns_this_large:
#         cost = global_cost_computation(symbol_flow, local_pattern, extended_s_and_c = extended_s_and_c, current_min = min_cost)
#         cost += irregularities_penalty(add, dele, penalty) # Penalty for the irregularities in the polytope.
#         if cost < min_cost:
#             min_cost = cost
#             best_pattern = local_pattern
#     return min_cost, best_pattern

# def optimal_pattern_on_polytope(symbol_flow, penalty = 1, measure = mvt.l1_norm, chromatic = False, extended_s_and_c = False):
#     # TODO: handle measures and chromatic
#     size = len(symbol_flow)
#     codes = pf.get_unique_codes(size) # Unique fo Guichaoua, not for extended s and c
#     if codes == []:
#         #raise err.NoPolytopeForThisSize("No Polytope is available for this size of segment. Try sequential cost instead.") from None
#         warnings.warn("No Polytope is available for this size of segment. Trying sequential cost instead.")
#         return sequential_score(symbol_flow, 0), "Sequential"
#     min_cost = math.inf
#     best_pattern = None
#     for add, dele in codes:
#         local_pattern = pf.make_polytope_pattern(int(round(math.log(size,2))), adding_code = add, deleting_code = dele)
#         cost = global_cost_computation(symbol_flow, local_pattern, extended_s_and_c = extended_s_and_c)
#         cost += irregularities_penalty(add, dele, penalty) # Penalty for the irregularities in the polytope.
#         if cost < min_cost:
#             min_cost = cost
#             best_pattern = local_pattern
#     return min_cost, best_pattern

