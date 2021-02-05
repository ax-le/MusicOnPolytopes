# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:50:39 2020

@author: amarmore

We constructed, in another file (``pattern_factory.py''), the patterns.
In this file, we will interpret the elements of this pattern, and develop means to access to each one of them individually.
Accessing to a specific element of the pattern can be made by indexing it.
The principle here will be based on binary number, as for irregularity codes, but with a different signification.
The goal is to see a n-dimension polytope as the nesting of two (n-1) dimension polytopes, and so on recursively.
Hence, indexing an element will be made by dichotomy.
A zero indicates to search in the "left" part of the pattern (so the first (n-1) dimension polytope), 
and a one indicates to search on the right part (so the second (n-1) dimension polytope).
In that sense, an element index needs to be of the size of the dimension of the pattern, 
with a special spectification for addition, which will be repsetned by tuples (as for addition in the pattern).
!! DISCLAIMER: irregularity codes and element indexes are not compatible and does not represent the same thing.

In this file, all patterns will have to be indexed.

References:
[1] Louboutin, C., & Bimbot, F. (2017, October). Modeling the multiscale structure of chord sequences using polytopic graphs.
[2] Guichaoua, C., & Bimbot, F. (2018, May). Inférence de segmentation structurelle par compression via des relations multi-échelles dans les séquences d'accords.

"""

import math
import copy

import polytopes.model.errors as err
import polytopes.data_manipulation as dm
import polytopes.pattern_factory as pf

# %% Element indexation      
def get_index_of_element(element, pattern):
    """
    Return the index of a specific element 'element' in an indexed pattern.
    
    This index is a boolean code, indicating how to navigate inside the pattern and the nesting of lists, by dichotomy.
    A zero indicates to search in the "left" part of the pattern, and a one indicates to search on the right part.
    In that sense, an element index will be of the size of the dimension of the pattern.
    Added elements will result in a tuple in the index (similar to tuples in the pattern).

    Parameters
    ----------
    element : integer
        The element to find in the pattern.
    pattern : nest list of integers (and tuples)
        The indexed pattern to search the element in.

    Returns
    -------
    list of binary numbers
        The index of this element, in the pattern.

    """
    return pf.flatten_nested_list(recursively_index_element(element, pattern))
    
def recursively_index_element(element, pattern):
    """
    Recursively computes the index of a specific element 'element' in an indexed pattern.
    
    This index is a boolean code, indicating how to navigate inside the pattern and the nesting of lists, by dichotomy.
    A zero indicates to search in the "left" part of the pattern, and a one indicates to search on the right part.
    In that sense, an element index will be of the size of the dimension of the pattern.
    Added elements will result in a tuple in the index (similar to tuples in the pattern).

    Parameters
    ----------
    element : integer
        The element to find in the pattern.
    pattern : nest list of integers (and tuples)
        The indexed pattern to search the element in.
        
    Raises
    ------
    ElementNotFound
        Error indicating that this element couldn't be found in this pattern.

    Returns
    -------
    list of binary numbers
        The index of this element, in the pattern.

    """
    if pf.get_pattern_dimension(pattern) == 1:
        if pattern[0] == element:
            return 0
        if type(pattern[0]) is tuple:
            if pattern[0][0] == element:
                return (0,0)
            elif pattern[0][1] == element:
                return (0,1)
        if len(pattern) != 1:
            if pattern[1] == element:   
                return 1
            if type(pattern[1]) is tuple:
                if pattern[1][0] == element:
                    return (1,0)
                if pattern[1][1] == element:
                    return (1,1)
        raise err.ElementNotFound("Element {} not found in the pattern {}.".format(element, pattern))
    else:
        if element in pf.flatten_pattern(pattern[0]): 
            tab = [0]
            tab.append(recursively_index_element(element, pattern[0]))
            return tab
        elif len(pattern) != 1:
            if element in pf.flatten_pattern(pattern[1]): 
                tab = [1]
                tab.append(recursively_index_element(element, pattern[1]))
                return tab
            else:
                raise err.ElementNotFound("Element {} not found in the pattern {}.".format(element, pattern))
        else:
            raise err.ElementNotFound("Element {} not found in the pattern {}.".format(element, pattern))
            
def get_element_with_index(index_element, pattern, with_tuples = False):
    """
    Return the element in the pattern, from its index.

    Parameters
    ----------
    index_element : list of binary numbers
        The index of the element to be found.
    pattern : list of nested integers
        The pattern, in which we will search for the element.
    with_tuples : boolean, optional
        A boolean, indicating if the element should be returned in a tuple or not.
        It only applies to elements which are in a tuple in a pattern,
        i.e. elements which are added or on which an element is added.
        If True, the entire tuple is returned and, if False, only the element corresponding to the index is returned. 
        This argument is mainly used for PPPs.
        The default is False.

    Raises
    ------
    InvalidIndexSizeException
        Error indicating that the index is incoherent with the pattern dimension.

    Returns
    -------
    integer or None
        Returns an integer (the element) if the index corresponds to an element in the pattern, or None otherwise.

    """
    if len(index_element) != pf.get_pattern_dimension(pattern):
        raise err.InvalidIndexSizeException("Wrong index {}, of different dimension than the pattern {}.".format(index_element, pattern))
    return recursively_find_element(index_element, pattern, with_tuples = with_tuples)
    
        
def recursively_find_element(idx_elt, pattern, with_tuples = False):
    """
    Recursively computes the element in the pattern, from its index.

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element to be found.
    pattern : list of nested integers
        The pattern, in which we will search for the element.
    with_tuples : boolean, optional
        A boolean, indicating if the element should be returned in a tuple or not.
        It only applies to elements which are in a tuple in a pattern,
        i.e. elements which are added or on which an element is added.
        If True, the entire tuple is returned and, if False, only the element corresponding to the index is returned. 
        This argument is mainly used for PPPs.
        The default is False.

    Raises
    ------
    InvalidIndexSizeException
        Error indicating that the index is incoherent with the pattern dimension.

    Returns
    -------
    integer or None
        Returns an integer (the element) if the index corresponds to an element in the pattern, or None otherwise.

    """
    if len(idx_elt) != pf.get_pattern_dimension(pattern):
        raise err.InvalidIndexSizeException("Wrong index {}, of different dimension than the pattern {}.".format(idx_elt, pattern))
    if pf.get_pattern_dimension(pattern) == 1:
        idx_elt = idx_elt[0]
        if type(idx_elt) is tuple:
            try:
                if type(pattern[idx_elt[0]]) is tuple:
                    return pattern[idx_elt[0]][idx_elt[1]]
                elif idx_elt[1] == 0:
                    return pattern[idx_elt[0]]
                else:
                    return None
                    #raise NotImplementedError("Index is a tuple {}, but the polytope doesn't have an addition {}.".format(idx_elt, pattern))
            except IndexError:
                return None
        else:
            try:
                if type(pattern[idx_elt]) is tuple:
                    if with_tuples:
                        return pattern[idx_elt]
                    else:
                        return pattern[idx_elt][0]
                return pattern[idx_elt]
            except IndexError:
                return None
    else:
        try:
            return recursively_find_element(idx_elt[1:], pattern[idx_elt[0]], with_tuples = with_tuples)
        except IndexError:
            return None
        
def delete_tuples(idx):
    """
    Replace the tuple in the index by its first element.
    
    Convenient when you're focusing on the first element of the tuple (the one originally present in the pattern), and working with its index.

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index to modify.

    Returns
    -------
    idx_cpy : list of binary numbers
        The index, without tuples (and with the first element of the tuple instead).

    """
    idx_cpy = copy.deepcopy(idx)
    if type(idx_cpy[-1]) is tuple:
        idx_cpy[-1] = idx_cpy[-1][0]
    return idx_cpy
    
def add_indexes(a, b):
    """
    Add two set of indexes, and raises errors accordingly (an index is binary, and cannot be composed of a "2" for instance).
    
    In that sense, it only works if both indexes never share a 1 at a same place.

    Parameters
    ----------
    a : list of binary numbers
        The index of the first element.
    b : list of binary numbers
        The index of the second element.

    Raises
    ------
    InvalidIndexSizeException
        An error, either if both indexed are of different sizes of if the sum results in a '2'.

    Returns
    -------
    to_return : list of binary numbers
        The index of the sum of both elements.

    """
    if len(a) != len(b):
        raise err.InvalidIndexSizeException("Both indexes ({} and {}) are of different lengths".format(a, b)) from None
    to_return = []
    if type(a[-1]) is tuple:
        for i in range(len(a) - 1):
            to_return.append(a[i] + b[i])
        if type(b[-1]) is tuple:
            tup = (a[-1][0] + b[-1][0], a[-1][1] + b[-1][1])
            if 2 in tup:
                raise err.InvalidIndexException("Summing indexes {} and {} resulted in a 2, should not happen.".format(a, b)) from None
            to_return.append(tup)
        else:
            to_return.append((a[-1][0] + b[-1], a[-1][1]))
            if a[-1][0] + b[-1] == 2:
                raise err.InvalidIndexException("Summing indexes {} and {} resulted in a 2, should not happen.".format(a, b)) from None
    elif type(b[-1]) is tuple:
        for i in range(len(a) - 1):
            to_return.append(a[i] + b[i])
        to_return.append((b[-1][0] + a[-1], b[-1][1]))
        if b[-1][0] + a[-1] == 2:
            raise err.InvalidIndexException("Summing indexes {} and {} resulted in a 2, should not happen.".format(a, b)) from None
    else:
        to_return = [a[i] + b[i] for i in range(len(a))]
    if 2 in to_return:
        raise err.InvalidIndexException("Summing indexes {} and {} resulted in a 2, should not happen.".format(a, b)) from None
    return to_return

# %% Antecedents and successors from elements
######## Antecedents
def get_antecedents_from_element(elt, pattern):
    """
    Return the antecedents (as elements) of this element (as element) in this indexed pattern.

    Parameters
    ----------
    elt : integer
        The element, whose antecedents are to be returned.
    pattern : list of nested integers
        The pattern, in which we will search for antecedents.

    Returns
    -------
    list of integers
        List of the antecedents, as integers.

    """
    idx_elt = get_index_of_element(elt, pattern)
    return get_antecedents_from_idx(idx_elt, pattern)

def get_antecedents_from_idx(idx_elt, pattern):
    """
    Return the antecedents (as elements) of this element (as index) in this indexed pattern.

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element, whose antecedents are to be returned.
    pattern : list of nested integers
        The pattern, in which we will search for antecedents.

    Returns
    -------
    list of integers
        List of the antecedents, as integers.

    """
    antecedents = get_antecedents_idx_from_idx(idx_elt)
    if antecedents == []:
        return []
    to_return = []
    for i in antecedents:
        ant = get_element_with_index(i, pattern, with_tuples = False)
        if ant != None:
            to_return.append(ant)
    return to_return
    
def get_antecedents_idx_from_idx(idx_elt):
    """
    Return the antecedents (as indexes) of this element (as index).
    
    This function does not take a pattern as argument, as the indexes of the antecedents does not depend on any.

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element, whose antecedents are to be returned.

    Returns
    -------
    list of list of binary numbers
        List of the antecedents, as element indexes.

    """
    if idx_elt == None:
        return []
    antecedents = []
    idx = copy.deepcopy(idx_elt)
    if type(idx[-1]) is tuple:
        if idx[-1][1] == 0:
            idx[-1] = idx[-1][0]
            return get_antecedents_idx_from_idx(idx)
        else:
            # # Without other addition as homologuous
            # idx[-1] = idx[-1][0]
            # return [idx]
            
            # With other addition as homologuous (no return, so it goes in the for loop)
            tab = idx[:-1]
            tab.append(idx[-1][0])
            antecedents.append(tab)
            if idx[-1][0] == 1:
                tab = idx[:-1]
                tab.append((0,1))
                antecedents.append(tab)
    for i in range(len(idx)):
        if idx[i] == 1:
            new_idx = copy.deepcopy(idx)
            new_idx[i] = 0
            antecedents.append(new_idx)                
    return antecedents

######## Pivot related to antecedents
def get_pivot_from_idx(elt_idx, ant_idx, pattern):
    """
    Returns the pivot (as element) of this element (elt_idx, as index)
    in relation with this antecedent (ant_idx, as index) in the pattern.

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element, for which we compute antecedent and pivot.
    ant_idx : list of binary numbers
        The index of the antecdent.
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Returns
    -------
    integer
        The pivot, as element (integer).

    """
    pivot_idx = get_pivot_idx_from_idx(elt_idx, ant_idx)
    return get_element_with_index(pivot_idx, pattern, with_tuples = False)

def get_pivot_idx_from_idx(elt_idx, ant_idx):
    """
    Returns the pivot (as index) of this element (elt_idx, as index) in relation with this antecedent (ant_idx, as index).

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element, for which we compute antecedent and pivot.
    ant_idx : list of binary numbers
        The index of the antecdent.

    Returns
    -------
    list of binary numbers
        The pivot, as index.
        
    """
    if type(elt_idx[-1]) == tuple:
        if elt_idx[-1][1] == 0:
            if type(ant_idx[-1]) != tuple:
                pivot_idx = [delete_tuples(elt_idx)[i] - ant_idx[i] for i in range(len(ant_idx))]
            elif elt_idx[-1][1] == ant_idx[-1][1]:
                pivot_idx = [delete_tuples(elt_idx)[i] - delete_tuples(ant_idx)[i] for i in range(len(ant_idx))]
            else: # Only difference is in last index, so its a warp, so return 0
                pivot_idx = [0 for i in range(len(elt_idx))]
        else:
            pivot_idx = [elt_idx[i] - ant_idx[i] for i in range(len(ant_idx) - 1)]
            if type(ant_idx[-1]) != tuple:
                if delete_tuples(elt_idx) != ant_idx:
                    raise NotImplementedError("That's an error, right? Is {} the antecedent of {}?".format(ant_idx, elt_idx))
                else:
                    pivot_idx.append(0)
            else:
                pivot_idx.append(elt_idx[-1][0] - ant_idx[-1][0])
    else:
        pivot_idx = [elt_idx[i] - ant_idx[i] for i in range(len(ant_idx))]
    return pivot_idx

def get_antecedents_with_pivots_from_idx(elt_idx, pattern):
    """
    Return a list of tuples (of integers), each tuple corresponding to a couple antecedents/pivot in relation with this antecedent (as elements).

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element, for which we compute antecedent and pivot.
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Returns
    -------
    list of tuples of integers
        Couples (antecedents, pivot) for this element.

    """
    antecedents_idx = get_antecedents_idx_from_idx(elt_idx)
    if antecedents_idx == []:
        return []
    else:
        this_elt_ant = []
        for ant_idx in antecedents_idx:
            ant = get_element_with_index(ant_idx, pattern, with_tuples = False)
            if ant != None:
                if ant == 0:
                    this_elt_ant.append((0,0))
                else:
                    pivot = get_pivot_from_idx(elt_idx, ant_idx, pattern)
                    this_elt_ant.append((ant, pivot))
        return this_elt_ant
    
def get_global_antecedents_with_pivots_from_idx(elt_idx, pattern):
    """
    Return a list of tuples (of integers), each tuple corresponding to a couple GLOBAL antecedents/pivot in relation with this antecedent (as elements).

    This function corresponds to a case in C. Guichaoua's framework, and is (for now) not used in my model.
    The principle is to consider antecedents not as the direct antecedents (elements linked by an arrow in the polytope),
    but as any element which can be linked to this element.
    TODO: Link to some of my reference.
    
    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element, for which we compute antecedent and pivot.
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Returns
    -------
    list of tuples of integers
        Couples (global antecedents, pivot) for this element.

    """
    antecedents_idx = get_antecedents_idx_from_idx(elt_idx)
    if antecedents_idx == []:
        return []
    elif [0 for i in range(len(elt_idx))] in antecedents_idx:
        return [(0,0)]
    for idx in antecedents_idx: # Antecedents of the antecedents and etc (the ``append'' is taken in account in the for loop, so it searches for the antecedents of the added antecedents)

        if idx != [0 for i in range(len(elt_idx))]:
            for ant_ant in get_antecedents_idx_from_idx(idx):
                if ant_ant not in antecedents_idx:
                    antecedents_idx.append(ant_ant)
    else:
        this_elt_ant = []
        for ant_idx in antecedents_idx:
            ant = get_element_with_index(ant_idx, pattern, with_tuples = False)
            if ant != None and ant != 0:
                try:
                    pivot = get_pivot_from_idx(elt_idx, ant_idx, pattern)
                    if (pivot, ant) not in this_elt_ant:
                        this_elt_ant.append((ant, pivot))
                except NotImplementedError:
                    pass
        return this_elt_ant

##### Successors of this element
def get_successors_from_element(elt, pattern):
    """
    Return the successors (as elements) of this element (as element), subject to this pattern (indexed).

    Parameters
    ----------
    elt : integer
        The element, for which we compute the successors.
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Returns
    -------
    list of integers
        The successors of this element (as elements).

    """
    idx_elt = get_index_of_element(elt, pattern)
    return get_successors_from_idx(idx_elt, pattern)

def get_successors_from_idx(idx_elt, pattern):
    """
    Return the successors (as elements) of this element (as index), subject to this pattern (indexed).

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element, for which we compute successors.
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.
        
    Returns
    -------
    list of integers
        The successors of this element (as elements).

    """
    successors = get_successors_idx_from_idx(idx_elt)
    if successors == []:
        return []
    to_return = []
    for i in successors:
        suc = get_element_with_index(i, pattern, with_tuples = False)
        if suc != None:
            to_return.append(suc)
    return to_return
    
def get_successors_idx_from_idx(idx_elt):
    """
    Return the successors (as indexes) of this element (as index).
    
    The returned successors are all the possible ones, and won't be present in all patterns.

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element, for which we compute successors.
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Returns
    -------
    list of list of binary numbers
        The successors of this element (as indexes).

    """
    if idx_elt == None:
        return []
    successors = []
    idx = copy.deepcopy(idx_elt)
    if type(idx[-1]) is tuple:
        if idx[-1][1] == 0:
            idx[-1] = idx[-1][0]
            successors = get_successors_idx_from_idx(idx)
            idx[-1] = (idx[-1], 1)
            successors.append(idx)
            return successors
        
        else:
            # # Without other addition as homologuous
            # return []
            
            # With other addition as homologuous (no return, so it goes in the for loop)
            if idx_elt[-1][0] == 0:
                tab = idx[:-1]
                tab.append((1,1))
                successors.append(tab)
    for i in range(len(idx)):
        if idx[i] == 0:
            new_idx = copy.deepcopy(idx)
            new_idx[i] = 1
            successors.append(new_idx)                
    return successors

def get_global_successors_from_idx(idx_elt, pattern):
    """
    Return the global successors (as elements) of this element (as index).
    
    The returned successors are all the possible ones, and won't be present in all patterns.
    This function corresponds to a case in C. Guichaoua's framework, and is (for now) not used in my model.
    The principle is to consider successors not as the direct successors (elements linked by an arrow in the polytope), 
    but as any element which can be linked to this element.
    TODO: Link to some of my reference.

    Parameters
    ----------
    idx_elt : list of binary numbers
        The index of the element, for which we compute successors.
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Returns
    -------
    list of integers
        The successors of this element (as element).

    """
    successors = get_successors_idx_from_idx(idx_elt) # Direct successors
    if successors == []:
        return []
    for idx in successors: # Successors of the successors and etc (the ``append'' is taken in account in the for loop, so it searches for the successors of the added successors)
        for suc_suc in get_successors_idx_from_idx(idx):
            if suc_suc not in successors:
                if type(suc_suc[-1]) is not tuple or suc_suc[-1][-1] != 1:
                    successors.append(suc_suc)
                elif type(idx_elt[-1]) is tuple and idx_elt[-1][-1] == 1:
                    successors.append(suc_suc)

    to_return = []
    for i in successors:
        suc = get_element_with_index(i, pattern, with_tuples = False)
        if suc != None:
            to_return.append(suc)
    return to_return

# %% Primers and under primers, for PPPs
def find_primers_of_low_level_systems(indexed_pattern):
    """
    Recursively find all primers (as elements) of low-level systems.
    
    This function is adapted for PPP primers retrieval.

    Parameters
    ----------
    indexed_pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Raises
    ------
    PatternToDebugError
        If the pattern is totally weird (of size different than 2 or 1, should never happen).

    Returns
    -------
    list of integers
        list of the primers of all low-level systems.

    """
    if pf.get_pattern_dimension(indexed_pattern) <= 2:
        return pf.flatten_pattern(indexed_pattern)[0]
    else:
        first_nested_pattern = indexed_pattern[0]
        size_dim_inf = pf.get_pattern_size(first_nested_pattern)
        primer_1 = find_primers_of_low_level_systems(first_nested_pattern)
        if len(indexed_pattern) == 2:
            primer_2 = find_primers_of_low_level_systems(indexed_pattern[1])
        elif len(indexed_pattern) != 1:
            raise err.PatternToDebugError("Pattern of invalid size: {}".format(indexed_pattern))
        else:
            return [find_primers_of_low_level_systems(indexed_pattern[0])]
        return [primer_1, primer_2]
    
def get_under_primers(pattern):
    """
    Return the elements (as indexed) which are successors of the primer (first element of the polytope).
    
    These are the elements which lead the permutations in the PPP model.
    Note: isn't that killing a fly with a bazooka? Under primers should be all elements containing a single one.
    (Could be tricky for polytope of dim n with n-1 dim irregularity, or for irregularity on under-primers though)

    Parameters
    ----------
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Returns
    -------
    under_primers : list of list of binary nimbers
        The under pirmers.

    """
    under_primers = []
    for i in range(1, pf.get_pattern_size(pattern)):
        idx_elt = get_index_of_element(i, pattern)
        if 0 in get_antecedents_from_idx(idx_elt, pattern):
            under_primers.append(idx_elt)
    return under_primers

# Generate PPPs
def generate_ppp(pattern):
    """
    Generate all PPPs (Primer Preserving Permutation) of this indexed pattern.

    Parameters
    ----------
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Returns
    -------
    all_ppps : TYPE
        DESCRIPTION.

    """
    all_ppps = []
    under_primers = get_under_primers(pattern)
    for fst_primer_list_idx, fst_primer in enumerate(under_primers):
        for snd_primer in under_primers[fst_primer_list_idx + 1:]:
            other_under_primers = [a for a in under_primers if a != fst_primer and a != snd_primer]
            fst_elt = [0 for i in range(len(fst_primer))]
            all_ppps.append(recursive_ppp(pattern, fst_primer, snd_primer, other_under_primers, fst_elt))
    return all_ppps

# Recursively generating PPPs
def recursive_ppp(pattern, fst_under_primer, snd_under_primer, other_under_primers, fst_elt):
    """
    Recursively computes a PPP, with 'fst_under_primer' and 'snd_under_primer' as the first two under primers (successors of the primer).

    Parameters
    ----------
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.
    fst_under_primer : list of binary numbers
        The first under primer defining the PPP.
    snd_under_primer : list of binary numbers
        The second under primer defining the PPP.
    other_under_primers : list of list of binary numbers
        The other under primers, for upper-level systems.
    fst_elt : integer
        Element useful for the recursion, determining the first element for half a part of the pattern.

    Returns
    -------
    nested list of integers
        A PPP for that pattern, determined by 'fst_under_primer' and 'snd_under_primer'.

    """
    if len(other_under_primers) == 0:
        left_ppp = ppp_dim_1_pattern(fst_elt, fst_under_primer, pattern)
        right_ppp = ppp_dim_1_pattern(add_indexes(fst_elt,snd_under_primer), fst_under_primer, pattern)
        while None in left_ppp:
            left_ppp.remove(None)
        while None in right_ppp:
            right_ppp.remove(None)
        if left_ppp == []:
            if right_ppp == []:
                return None
            else:
                return [right_ppp]
        elif right_ppp == []:
            return [left_ppp]
        else:
            return [left_ppp, right_ppp]
    else:
        left_ppp = recursive_ppp(pattern, fst_under_primer, snd_under_primer, other_under_primers[:-1], fst_elt)
        right_ppp = recursive_ppp(pattern, fst_under_primer, snd_under_primer, other_under_primers[:-1],add_indexes(other_under_primers[-1], fst_elt))
        if left_ppp == None:
            if right_ppp == None:
                return None
            else:
                return [right_ppp]
        elif right_ppp == None:
            return [left_ppp]
        else:
            return [left_ppp, right_ppp]
        
def ppp_dim_1_pattern(first_elt_idx, up_idx, pattern):
    """
    Compute the pattern composed of 'first_elt_idx' and the successor of 'first_elt_idx' which has up_idx as pivot.
    
    In that sense, this pattern is homologuous to the relation between the primer and the under primer up_idx, and starts at first_elt_idx.
    (This function is actually the reason for the argument 'with_tuples', as we want to keep addition here)

    Parameters
    ----------
    first_elt_idx : list of binary numbers
        The first element for this pattern.
    up_idx : list of binary numbers
        The under primer which relation with the primer we want to apply on 'first_elt_idx'.
    pattern : list of nested integers
        The pattern, in which we will search for the pivot, as element.

    Returns
    -------
    list (couple) of integers or tuples
        A PPP of 1 dimension, composed of 'first_elt_idx' and its successor, determined by the relation between the primer and 'up_idx'.

    """
    first_elt = get_element_with_index(first_elt_idx, pattern, with_tuples = True)
    snd_elt = get_element_with_index(add_indexes(first_elt_idx, up_idx), pattern, with_tuples = True)
    return [first_elt, snd_elt]

def swap_chord_sequence(chord_sequence, permuted_elements_list):
    """
    Swap the chord sequence according to the current PPP.
    
    In that sense, the permuted chord sequence will follow the permuted order of the current PPP.

    Parameters
    ----------
    chord_sequence : list
        List of Chords, of any type.
    permuted_elements_list : list of integers
        The elements of the permuted indexed pattern.

    Returns
    -------
    new_chord_seq : list
        Permuted list of Chords.

    """
    new_chord_seq = []
    for i in permuted_elements_list:
        new_chord_seq.append(chord_sequence[i])
    return new_chord_seq

def compute_direct_antecedents(indexed_pattern):
    """
    Compute the direct antecedent of every element of the pattern as described by Louboutin's paradigm.
    
    Indeed, each element of the Louboutin's paradigm has only one antecedent, which is conditionned to the current PPP, and can be a fictive element.

    Parameters
    ----------
    indexed_pattern : list of nested integers
        The pattern, which can be permuted in a PPP framework.

    Returns
    -------
    direct_antecedents : list of integers or tuples of three integers
        The list of direct antecedents.
        When the antecedent is a tuple, it means that its antecedent will be the fictive element constructed by this system.

    """
    pattern_size = pf.get_pattern_size(indexed_pattern)
    direct_antecedents = [None for i in range(pattern_size)]
    for i in range(1, pattern_size):
        direct_antecedents[i] = recursively_find_direct_antecedent(i, indexed_pattern)
    return direct_antecedents

def recursively_find_direct_antecedent(i, indexed_pattern):
    """
    Find the direct antecedent of element 'i' in the current indexed pattern.
    
    This direct antecedent is the one defined by C. Louboutin's framework, and can be a fictive element.
    When it is a fictive element, we return a tuple with the 3 elements defining the system.

    Parameters
    ----------
    i : element
        The element, whose antecedent is to compute.
    indexed_pattern : list of nested integers
        The pattern, which can be permuted in a PPP framework.

    Raises
    ------
    InvalidIndexException
        In cases of invalid indexes.

    Returns
    -------
    integer or tuple of three integers
        The antecedent, or the three elements to construct a system with (as the antecedent will be the fictive element determined by this system).

    """
    elt_idx = get_index_of_element(i, indexed_pattern)
    if type(elt_idx[-1]) is tuple:
        if elt_idx[-1][1] == 1:
            return i-1 # Previous element, on which it is attached
        else:
            elt_idx[-1] = elt_idx[-1][0]

    pattern_dim = pf.get_pattern_dimension(indexed_pattern)
    if pattern_dim < 2:
        if len(elt_idx) != 1:
            raise err.InvalidIndexException("Index of invalid size: {} (should be of size 1).".format(elt_idx))
        else:
            if elt_idx[0] == 0:
                raise err.InvalidIndexException("Cannot compute an antecedent as it is a primer (index of 0): {}.".format(elt_idx))
            elif type(indexed_pattern[0]) is tuple:
                return indexed_pattern[0][0]
            else:
                return indexed_pattern[0]
        
    # By construction, whatever the PPP is, the index will always return a same POSITION and not ELEMENT.
    # In that sense, both under-primers used for low-level system definition will always be [0*]01 and [0*]10
    up_one = [0 for i in range(pattern_dim)]
    up_one[-1] = 1

    up_two = [0 for i in range(pattern_dim)]
    up_two[-2] = 1
    if elt_idx.count(1) == 1: # It's an under-primer
        return 0
    else:
        elt_minus_up_one_idx = [elt_idx[i] - up_one[i] for i in range(pattern_dim)]
        elt_minus_up_two_idx = [elt_idx[i] - up_two[i] for i in range(pattern_dim)]
        
        if -1 in elt_minus_up_one_idx:
            if -1 in elt_minus_up_two_idx: # Is an element of high-level systems
                primers_pattern = find_primers_of_low_level_systems(indexed_pattern)
                return recursively_find_direct_antecedent(i, primers_pattern)
            else:
                return get_element_with_index(elt_minus_up_two_idx, indexed_pattern)
        else:
            if -1 in elt_minus_up_two_idx:
                return get_element_with_index(elt_minus_up_one_idx, indexed_pattern)
            else:
                local_primer_idx = [elt_idx[i] - up_one[i] - up_two[i] for i in range(pattern_dim)]
                local_primer = get_element_with_index(local_primer_idx, indexed_pattern) # a_1
                elt_minus_up_one = get_element_with_index(elt_minus_up_one_idx, indexed_pattern) # a_2
                elt_minus_up_two = get_element_with_index(elt_minus_up_two_idx, indexed_pattern) # a_3
                return (local_primer, elt_minus_up_one, elt_minus_up_two)
       
