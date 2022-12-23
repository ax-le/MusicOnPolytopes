# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:22:23 2021

@author: amarmore

# Generating polytopes: ``Patterns''.
We call in this part ``pattern'' the nested/computationnal representation of a polytope.
This work relies mainly on studies operated by Corentin Louboutin [1] and Corentin Guichaoua [2].

The principle of polytopes is to adapt chord sequences to geometrical forms, and infer relations between these elements with geomtrical proximity.
Hence, musical elements won't be (only) linked in chronological order, but may be linked by their inner position in different bars.
``Patterns'' computationnaly represent polytopes. Concretely, they are nested list.
The idea of nesting list comes from the fact that polytopes are n-hypercubes, 
and that a n-hypercube can be seen as the concatenation of 2 (n-1)-hypercubes.
Hence, recursively, we can nest any n-hypercube as a concatenation of some m-hypercbes (m < n, 2**(n-m) to be precise).
TODO: Add own reference when published/arXived, comments won't be enough.

This code below is intended to generate the patterns, which will then be used for intepreting music.
Basically, polytopes are either n-dimensional hypercubes (called ``regular'' polytopes),
or n-dimensions hypercubes where m-dimensional hypercubes have been deleted and/or added (m < n).
More details can be found in [2].
The idea is then to generate a n-dimensional hypercube, and then to alter it.
This is made by encoding the positions which need to be altered, either by addition or deletion.
In this file, they're called addition or deletion "codes", and are lists of booleans (0 or 1).
Concretely, all codes are binary numbers, and are constructed to indicate where and how many positions need to be altered.
Even though each element in the polytope can be specified by dichotomy (by construction, by concatenating two polytopes when increasing the dimension, see pattern_manip.py),
**the goal of codes here is not to indicate individually each position which need to be altered**. 

- Codes are related to the notion of dimensions in the nesting:
     - The first element of the code (left element, the first in the list) will encode information about the 2 polytopes of dimension n-1
     - The second element of the code will encode information about polytopes at the n-2 dimension
     - The third element of the code will encode information about polytopes at the n-3 dimension
     - And so on until the last one, which represent information at the last nesting dimension (so directly on elements)
     
- As a dichotomy principle, the alteration will be propagated with binary rules at each dimension:
     - If the current boolean is a 1, this alteration will affect both nested polytopes.
     - If the current boolean is a 0, this alteration will only affect the second nested polytope, the one "on the right" (geometrically).

- In that sense, at every dimension, if the current boolean is 0, the code will only be propagated to the 2nd nested polytope, and the 1st nested polytope will be left without alteration.
Otherwise, if it's a 1, the rest of the code will be copied to both polytopes of lower dimension.

- At the last level, a 0 will indicate to alter only the 2nd polytope (so the element on the right), and a 1 will indicate to alter both.
Hence, a code composed of zeroes ([0,0,...,0]) will still alter the last element! To specify "no alteration at all", code must be an empty list.

In that sense, by dichotomy, we specify, for each dimension, to which part of the nesting alteration should be propagated (both or only the second part).
As it is a dichotomy principle, codes have to be of the same length than the dimension of the polytope.

In addition, the number of "1" in the code define the dimension of the alteration polytope.

To this instant, I don't have reference with figures to illustrate this principle (apart from the Tutorial Notebook, on the associated folder),
but the construction of irregular polytopes can be better understood by looking at function 'recursive_irregular_construction()'
    
Patterns are present under two forms : 
    - patterns of ``1''s : each element in the pattern will be a one, in order to represent only the general form of this pattern.
    - indexed pattern : a pattern where each element is represented by the index of this element.
TODO: find better names for these objects.

### References:
[1] Louboutin, C., & Bimbot, F. (2017, October). Modeling the multiscale structure of chord sequences using polytopic graphs.
[2] Guichaoua, C., & Bimbot, F. (2018, May). Inférence de segmentation structurelle par compression via des relations multi-échelles dans les séquences d'accords.
"""

import math
import numpy as np

import polytopes.data_manipulation as dm
import polytopes.model.errors as err

# Irregularities codes
def generate_irreg_codes(dimension, max_irregular_dimension):
    """
    Generate codes for irregular polytopes/patterns.
    
    Generating irregular patterns is made by indexing positions which need to be altered.
    This is the sense of this function, which will generate all possible codes given a polytope dimension and a maximal deformation dimension.
    Concretely, all codes are binary numbers, and are constructed to indicate where and how many positions need to be altered.
    Even though each element in the polytope can be specified by dichotomy (by construction, by concatenating two polytopes when increasing the dimension, see pattern_manip.py),
    **the goal of codes here is not to indicate individually each position which need to be altered**. 
    
    - Codes are related to the notion of dimensions in the nesting:
         - The first element of the code (left element, the first in the list) will encode information about the 2 polytopes of dimension n-1
         - The second element of the code will encode information about polytopes at the n-2 dimension
         - The third element of the code will encode information about polytopes at the n-3 dimension
         - And so on until the last one, which represent information at the last nesting dimension (so directly on elements)
         
    - As a dichotomy principle, the alteration will be propagated with binary rules at each dimension:
         - If the current boolean is a 1, this alteration will affect both nested polytopes.
         - If the current boolean is a 0, this alteration will only affect the second nested polytope, the one "on the right" (geometrically).
    
    - In that sense, at every dimension, if the current boolean is 0, the code will only be propagated to the 2nd nested polytope, and the 1st nested polytope will be left without alteration.
    Otherwise, if it's a 1, the rest of the code will be copied to both polytopes of lower dimension.
    
    - At the last level, a 0 will indicate to alter only the 2nd polytope (so the element on the right), and a 1 will indicate to alter both.
    Hence, a code composed of zeroes ([0,0,...,0]) will still alter the last element! To specify "no alteration at all", code must be an empty list.
    
    In that sense, by dichotomy, we specify, for each dimension, to which part of the nesting alteration should be propagated (both or only the second part).
    As it is a dichotomy principle, codes have to be of the same length than the dimension of the polytope.
    
    In addition, the number of "1" in the code define the dimension of the alteration polytope.
    
    To this instant, I don't have reference with figures to illustrate this principle (apart from the Tutorial Notebook, on the associated folder),
    but the construction of irregular polytopes can be better understood by looking at function 'recursive_irregular_construction()'
    
    Parameters
    ----------
    dimension : integer
        The dimension of the regular polytope.
        This dimension is also the size of the binary code.
    max_irregular_dimension : integer
        The maximal dimension of the irregulairty (included).
        By contruction, max_irregular_dimension < dim, and, in general, max_irregular_dimension = dim - 2.

    Returns
    -------
    to_return : list of binary numbers (as list)
        Codes which will generate all possible irregularities of given size.

    """
    form = "{0:0"+str(dimension)+"b}"
    to_return = [[]]
    for i in range(2**dimension):
        binary = form.format(i)
        if binary.count('1') <= max_irregular_dimension:
            to_add = []
            [to_add.append(int(i)) for i in binary]
            to_return.append(to_add)
    return to_return

def get_codes(desired_size):
    """
    List all couple of codes (addition and deletion) which will result in an irregular polytope of size 'desired_size'.
    
    This function mainly comes from C. Guichoua's work and code.

    Parameters
    ----------
    desired_size : integer
        The desired_size for a polytope.

    Returns
    -------
    to_return : list of binary numbers (as list)
        List of all possible couple of codes which will result in a pattern of size 'desired_size'.

    """
    dim = round(math.log(desired_size,2))
    to_return = []
    adding_codes_all = generate_irreg_codes(dim, dim - 2)
    deleting_codes_all = generate_irreg_codes(dim, dim - 2)
    for adding_code in adding_codes_all:
        for deleting_code in deleting_codes_all:
            if get_final_pattern_size(dim, adding_code, deleting_code) == desired_size:
                to_return.append((adding_code, deleting_code))
    return to_return

def get_unique_codes(desired_size):
    """
    List all couple of codes (addition and deletion) which will result in uniques irregular polytope of size 'desired_size'.
    
    Parameters
    ----------
    desired_size : integer
        The desired_size for a polytope.

    Returns
    -------
    unique_codes : list of binary numbers (as list)
        List of all possible couple of codes which will result in a unique pattern of size 'desired_size'.

    """
    already_found = []
    unique_codes = []
    for add, dele in get_codes(desired_size):
        patt = make_polytope_pattern(round(math.log(desired_size,2)), adding_code = add, deleting_code = dele)
        # Previous line will also construct regular patterns if size is 2**n
        if patt not in already_found:
            already_found.append(patt)
            unique_codes.append((add, dele))
    return unique_codes

def get_final_pattern_size(dimension, adding_code, deleting_code):
    """
    Compute the size of the pattern which will be generated with these codes.

    Parameters
    ----------
    dimension : integer
        Dimension of the regular pattern.
    adding_code : binary number (as list of 0 and 1)
        The irregular code for addition.
    deleting_code : binary number (as list of 0 and 1)
        The irregular code for deletion.

    Raises
    ------
    WrongIrregularCode
        Error indicating that the irregular codes are invalid.

    Returns
    -------
    integer
        The size of the final pattern.

    """
    regular = 2**dimension
    if adding_code == []:
        return regular - get_deformation_size(deleting_code)
    elif deleting_code == []:
        return regular + get_deformation_size(adding_code)
    else:
        if len(adding_code) != len(deleting_code):
            raise err.WrongIrregularCode("Adding and deleting codes must be of same size")
        overlap_dimension = 0
        for index in range(dimension):
            if deleting_code[index] == 1 and adding_code[index] == 1:
                    overlap_dimension += 1
        return regular + get_deformation_size(adding_code) - get_deformation_size(deleting_code) - 2**overlap_dimension
    
def get_deformation_size(irr_code):
    """
    Compute the size of the deformation generated by this code.

    Parameters
    ----------
    irr_code : binary number (as list of 0 and 1)
        The irregular code.

    Returns
    -------
    integer
        The size of the deformation induced by this code.

    """
    if irr_code == []:
        return 0
    return 2**irr_code.count(1)

def make_polytope_pattern(dimension, adding_code, deleting_code):
    """
    Generate the pattern, with his dimension (for the regular basis), and addition/deletion codes.

    Parameters
    ----------
    dimension : integer
        The dimension of the pattern.
    adding_code : binary number (as list of 0 and 1)
        The irregular code for addition.
    deleting_code : binary number (as list of 0 and 1)
        The irregular code for deletion.

    Raises
    ------
    WrongIrregularCode
        An error indicating that the irregular code is invalid.

    Returns
    -------
    nested list
        The pattern.

    """
    if len(adding_code) != dimension and len(adding_code) != 0:
        raise err.WrongIrregularCode("Wrong adding_code code") from None
    if len(deleting_code) != dimension and len(deleting_code) != 0:
        raise err.WrongIrregularCode("Wrong deleting_code code") from None
    if adding_code == [] and deleting_code == []:
        return make_regular_polytope_pattern(dimension)
    return recursive_irregular_construction(dimension, adding_code, deleting_code)

def make_regular_polytope_pattern(dimension):
    """
    Generate the regular pattern of dimension 'dimension'.

    Parameters
    ----------
    dimension : integer
        Dimension of the polytope.

    Returns
    -------
    nested list
        The pattern.

    """
    return recursive_regular_construction(dimension)

def recursive_regular_construction(dimension):
    """
    Recursively construct a nested list of 1s, representing a regular pattern of dimension 'dim'.

    Parameters
    ----------
    dimension : integer
        The dimension of this pattern.

    Returns
    -------
    1 or a nested list of 1
        The pattern of given dimension.

    """
    if dimension == 0:
        return 1
    return [recursive_regular_construction(dimension - 1) for i in range(2)]
    
def recursive_irregular_construction(dimension, adding_code, deleting_code):
    """
    Recursively construct nested lists of 1, representing an irregular pattern.
    
    Concretely, all codes are binary numbers, and are constructed to indicate where and how many positions need to be altered.
    Firstly, the last element of the polytope has to be altered when there is alteration. This is a construction constraint.
    Hence, a code entirely composed of zeroes ([0,0,...,0]) will still alter the last element.
    Codes are related to the notion of dimensions in the nesting:
     - The first element of the code (left element, the first in the list) will encode information about the 2 polytopes of dimension n-1
     - The second element of the code will encode information about polytopes at the n-2 dimension
     - The third element of the code will encode information about polytopes at the n-3 dimension
     - And so on until the last one, which represent information at the last nesting dimension (so elementary lines)
    A 1 in a code indicates that alteration is supposed to happen on both nested polytopes. So, recursively, the rest of the code will be copied to both polytopes of lower dimension.
    A 0 in the code indicates that only the second lower-dimension nested pattern will be altered, and not the first (left) part of the polytope.

    Parameters
    ----------
    dimension : integer
        The dimension of the pattern.
    adding_code : binary number (as list of 0 and 1)
        The irregular code for addition.
    deleting_code : binary number (as list of 0 and 1)
        The irregular code for deletion.

    Returns
    -------
    list or nested list
        The irregular pattern given the codes and its dimension.

    """
    if dimension == 1:
        if len(deleting_code) == 0:
            if len(adding_code) == 0:
                return [1,1]
            elif adding_code[0] == 1:
                return [(1,1),(1,1)]
            else:
                return [1, (1,1)]
        elif deleting_code[0] == 1:
            return []
        else:
            if len(adding_code) != 0 and adding_code[0] == 1:
                return [(1,1)]
            else:
                return [1]
    else:
        child = [0, 0]
        if len(deleting_code) != 0 and deleting_code[0] == 1:
            if len(adding_code) != 0 and adding_code[0] == 1:
                child[0] = recursive_irregular_construction(dimension - 1, adding_code[1:], deleting_code[1:])
            else:
                child[0] = recursive_irregular_construction(dimension - 1, [], deleting_code[1:])
        else:
            if len(adding_code) != 0 and adding_code[0] == 1:
                child[0] = recursive_irregular_construction(dimension - 1, adding_code[1:], [])
            else:
                child[0] = recursive_irregular_construction(dimension - 1, [], [])

        child[1] = recursive_irregular_construction(dimension - 1, adding_code[1:], deleting_code[1:])
        
        final_child = [x for x in child if x != []]
        return final_child
    
def make_indexed_pattern(dimension, adding_code, deleting_code, starting_index = 0):
    """
    Generate a pattern, where its elements are the indexes of these element in the temporal order.
    
    In that sense, each element of the pattern will represent the index of the chord (in the chord sequence) to evaluate.

    Parameters
    ----------
    dimension : integer
        The dimension of the pattern.
    adding_code : binary number (as list of 0 and 1)
        The irregular code for addition.
    deleting_code : binary number (as list of 0 and 1)
        The irregular code for deletion.
    starting_index : integer, optional
        The index for the first element. The default is 0.

    Raises
    ------
    WrongIrregularCode
        An error indicating that the irregular code is invalid.

    Returns
    -------
    nested list of integers
        The pattern, where elements are indexes.

    """
    if len(adding_code) != dimension and len(adding_code) != 0:
        raise err.WrongIrregularCode("Wrong adding_code code") from None
    if len(deleting_code) != dimension and len(deleting_code) != 0:
        raise err.WrongIrregularCode("Wrong deleting_code code") from None
    return recursive_pattern_indexed(dim = dimension, adding_code = adding_code, deleting_code = deleting_code, idx = starting_index)
    
def recursive_pattern_indexed(dim, adding_code, deleting_code, idx):
    """
    Recursive function to generate the indexed pattern.
    
    Concretely, all codes are binary numbers, and are constructed to indicate where and how many positions need to be altered.
    Firstly, the last element of the polytope has to be altered when there is alteration. This is a construction constraint.
    Hence, a code entirely composed of zeroes ([0,0,...,0]) will still alter the last element.
    Codes are related to the notion of dimensions in the nesting:
     - The first element of the code (left element, the first in the list) will encode information about the 2 polytopes of dimension n-1
     - The second element of the code will encode information about polytopes at the n-2 dimension
     - The third element of the code will encode information about polytopes at the n-3 dimension
     - And so on until the last one, which represent information at the last nesting dimension (so elementary lines)
    A 1 in a code indicates that alteration is supposed to happen on both nested polytopes. So, recursively, the rest of the code will be copied to both polytopes of lower dimension.
    A 0 in the code indicates that only the second lower-dimension nested pattern will be altered, and not the first (left) part of the polytope.

    Parameters
    ----------
    dim : integer
        The dimension of the pattern.
    adding_code : binary number (as list of 0 and 1)
        The irregular code for addition.
    deleting_code : binary number (as list of 0 and 1)
        The irregular code for deletion.
    idx : integer
        The index for the first element.

    Returns
    -------
    list or nested list of integers
        The indexed pattern.

    """
    if dim == 1:
        if len(deleting_code) == 0:
            if len(adding_code) == 0:
                return [idx, idx + 1]
            elif len(adding_code) != 0 and adding_code[0] == 1:
                return [(idx,idx + 1),(idx + 2,idx + 3)]
            else:
                return [idx, (idx + 1,idx + 2)]
        elif deleting_code[0] == 1:
            return []
        else:
            if len(adding_code) != 0 and adding_code[0] == 1:
                return [(idx,idx + 1)]
            else:
                return [idx]
    else:
        child = [0, 0]
        if len(deleting_code) != 0 and deleting_code[0] == 1:
            if len(adding_code) != 0 and adding_code[0] == 1:
                child[0] = recursive_pattern_indexed(dim - 1, adding_code[1:], deleting_code[1:], idx)
                size_first_poly = get_final_pattern_size(dim - 1, adding_code[1:], deleting_code[1:])
                child[1] = recursive_pattern_indexed(dim - 1, adding_code[1:], deleting_code[1:], idx + size_first_poly)
            else:
                child[0] = recursive_pattern_indexed(dim - 1, [], deleting_code[1:], idx)
                size_first_poly = get_final_pattern_size(dim - 1, [], deleting_code[1:])
                child[1] = recursive_pattern_indexed(dim - 1, adding_code[1:], deleting_code[1:], idx + size_first_poly)
        else:
            if len(adding_code) != 0 and adding_code[0] == 1:
                child[0] = recursive_pattern_indexed(dim - 1, adding_code[1:], [], idx)
                size_first_poly = get_final_pattern_size(dim - 1, adding_code[1:], [])
                child[1] = recursive_pattern_indexed(dim - 1, adding_code[1:], deleting_code[1:], idx + size_first_poly)
            else:
                child[0] = recursive_pattern_indexed(dim - 1, [], [], idx)
                child[1] = recursive_pattern_indexed(dim - 1, adding_code[1:], deleting_code[1:], idx + 2 ** (dim - 1))
        final_child = [x for x in child if x != []]

        return final_child

def index_this_pattern(pattern, starting_index = 0):
    """
    Return an indexed pattern from a 1s pattern.

    Parameters
    ----------
    pattern : nested list of 1s
        The pattern, where elements are all 1.
    starting_index : integer
        The index for the first element of the pattern.

    Returns
    -------
    nested list of integers
        The indexed pattern, corresponding to the pattern of 1.

    """
    return recursive_index_pattern(pattern,starting_index)

def recursive_index_pattern(pattern, idx):
    """
    Recursively generates the indexed version of a 1s pattern.

    Parameters
    ----------
    pattern : nested list of 1s
        The pattern, where elements are all 1.
    idx : integer
        The index for the first element of the pattern.

    Returns
    -------
    nested list of integers
        The indexed pattern, corresponding to the pattern of 1.

    """
    if get_pattern_dimension(pattern) == 1:
        to_return = []
        current_idx = idx
        for i in pattern:
            if type(i) is tuple:
                to_return.append((current_idx, current_idx + 1))
                current_idx += 2
            else:
                to_return.append(current_idx)
                current_idx += 1
        return to_return
    else:
        if len(pattern) == 1:
            return [recursive_index_pattern(pattern[0], idx)]
        else:
            size = get_pattern_size(pattern[0])
            return [recursive_index_pattern(pattern[0], idx), recursive_index_pattern(pattern[1], idx + size)]
    
def apply_chords_on_pattern(pattern,chord_sequence):
    """
    Replace the elements of the pattern with the elements of the chord sequence (major or minor triads).
    
    This function is only used for visualization purposes (in my code at least).

    Parameters
    ----------
    pattern : nest list of 1 or of integers
        The pattern (of 1 or indexed) to apply chords on.
    chord_sequence : list of string or Chord objects
        The chords (string or Chord object, see Chord.py) to apply on the pattern.

    Returns
    -------
    nested list of strings 
        The chords, as major or minor triads, nested to form the pattern.

    """
    if len(chord_sequence) != get_pattern_size(pattern):
        err.PatternAndSequenceIncompatible("The chord sequence and the pattern are of different sizes.")
    symbolic_chord_sequence = []
    for chord in chord_sequence:
        if dm.is_a_chord_object(chord):
            symbolic_chord_sequence.append(chord.triad)
        else:
            symbolic_chord_sequence.append(chord)
    if is_indexed_pattern(pattern):
        return recursive_chords_on_indexed_pattern(pattern,symbolic_chord_sequence)
    else:
        return recursive_chords_on_pattern(pattern,symbolic_chord_sequence)
    
def recursive_chords_on_pattern(pattern,chord_sequence):
    """
    Recursively replace the elements of the pattern with the elements of the chord sequence.
    
    Only works on pattern of 1s.
    (To be precise, it works on indexed patterns BUT doesn't use the indexes of the indexed pattern. In that sense, it doesn't work when indexes are not in the temporal order (PPPs for example).)

    Parameters
    ----------
    pattern : nested list of 1
        The pattern of 1 to apply chords on.
    chord_sequence : list of string or Chord objects
        The chords (string or Chord object, see Chord.py) to apply on the pattern.

    Returns
    -------
    nested list of strings 
        The chords, as major or minor triads, nested to form the pattern.

    """
    if get_pattern_dimension(pattern) == 1:
        to_return = []
        current_idx = 0
        for i in pattern:
            if type(i) is tuple:
                to_return.append((chord_sequence[current_idx], chord_sequence[current_idx + 1]))
                current_idx += 2
            else:
                to_return.append(chord_sequence[current_idx])
                current_idx += 1
        return to_return
    else:
        if len(pattern) == 1:
            return [recursive_chords_on_pattern(pattern[0],chord_sequence)]
        else:
            size = get_pattern_size(pattern[0])
            return [recursive_chords_on_pattern(pattern[0],chord_sequence[:size]), recursive_chords_on_pattern(pattern[1], chord_sequence[size:])]    

def recursive_chords_on_indexed_pattern(indexed_pattern,chord_sequence):
    """
    Recursively replace the elements of the pattern with the elements of the chord sequence.
    
    Only works on indexed patterns (and has been specifically made for PPPs).

    Parameters
    ----------
    pattern : nested list of integers
        The pattern of integers to apply chords on.
    chord_sequence : list of string or Chord objects
        The chords (string or Chord object, see Chord.py) to apply on the pattern.

    Returns
    -------
    nested list of strings 
        The chords, as major or minor triads, nested to form the pattern.

    """
    flattened = flatten_pattern(indexed_pattern)
    if get_pattern_dimension(indexed_pattern) == 1:
        to_return = []
        current_idx = 0
        for i in indexed_pattern:
            if type(i) is tuple:
                to_return.append((chord_sequence[flattened[current_idx]], chord_sequence[flattened[current_idx + 1]]))
                current_idx += 2
            else:
                to_return.append(chord_sequence[flattened[current_idx]])
                current_idx += 1
        return to_return
    else:
        if len(indexed_pattern) == 1:
            return [recursive_chords_on_indexed_pattern(indexed_pattern[0],chord_sequence)]
        else:
            return [recursive_chords_on_indexed_pattern(indexed_pattern[0],chord_sequence), recursive_chords_on_indexed_pattern(indexed_pattern[1], chord_sequence)]    

############################### Pattern information retrieval (size, dimension, etc)
def get_pattern_size(pattern):
    """
    Return the number of elements in the pattern.

    Parameters
    ----------
    pattern : nest list of 1 or of integers
        The pattern (of 1 or indexed) to evaluate.

    Returns
    -------
    size : integer
        The number of elements of this pattern (its size).

    """
    size = 0
    for i in pattern:
        if type(i) is int:
            size += 1
        elif type(i) is list or type(i) is tuple:
            size += get_pattern_size(i)
    return size

def get_pattern_dimension(pattern):
    """
    Return the dimension of the pattern.

    Parameters
    ----------
    pattern : nest list of 1 or of integers
        The pattern (of 1 or indexed) to evaluate.

    Returns
    -------
    dim : integer
        The dimension of this pattern.

    """
    dim = 1
    nested = pattern[0]
    while isinstance(nested, list):
        dim += 1
        nested = nested[0]
    return dim

def flatten_pattern(pattern):
    """
    Neutralize all nested lists and tuples, and returns the elements as a list.
    
    DISCLAIMER: tuples are neutralized in this function.
    To keep tuples, use: 'flatten_nested_list(pattern)'.

    Parameters
    ----------
    pattern : nested list of 1 or integers, and tuples
        The pattern, to flatten.

    Returns
    -------
    list of 1 or integer
        Elements of the pattern, as a list.

    """
    if type(pattern) is not list and type(pattern) is not tuple:
        return [pattern]
    else:
        to_return = flatten_pattern(pattern[0])
        for nested in pattern[1:]:
            to_return.extend(flatten_pattern(nested))
        return to_return
    
def flatten_nested_list(this_list):
    """
    Neutralize all nested lists, and returns the elements.
    
    DISCLAIMER: tuples are not lists, and are returned with this function.
    To neutralize tuples, use: 'flatten_pattern(pattern)'.

    Parameters
    ----------
    this_list : nested list of 1 or integers, and tuples
        The list, to flatten.

    Returns
    -------
    list of 1 or integers, and tuples
        Elements of this list, as a list. Tuples remain tuples.s

    """
    if type(this_list) is not list:
        return [this_list]
    else:
        to_return = flatten_nested_list(this_list[0])
        for nested in this_list[1:]:
            to_return.extend(flatten_nested_list(nested))
        return to_return
    
def is_indexed_pattern(pattern):
    """
    Check if this pattern is indexed, or not (a pattern of 1).

    Parameters
    ----------
    pattern : nested list of 1 or integers
        The pattern, to evaluate.

    Returns
    -------
    boolean
        True if this pattern is indexed, False otherwise.

    """
    if get_pattern_size(pattern) < 2:
        raise err.NoPolytopeForThisSize("No polytope should be of size 1.")
    if len(np.unique(flatten_pattern(pattern))) not in [1, get_pattern_size(pattern)]:
        raise err.PatternToDebugError("This pattern seems to have repetition in its elements, but isn't a pattern of ones, to deubg: {}.".format(pattern))
    return len(np.unique(flatten_pattern(pattern))) != 1

def extract_pattern_from_indexed_pattern(indexed_pattern):
    """
    Return a pattern of 1 from an indexed pattern.

    Parameters
    ----------
    indexed_pattern : nested list of integers
        The indexed pattern, to evaluate.

    Raises
    ------
    PatternToDebugError
        Errors in case of illicit pattern.

    Returns
    -------
     nested list of 1
        The pattern of one, extracted from the indexed pattern.

    """
    if get_pattern_dimension(indexed_pattern) == 1:
        if len(indexed_pattern) == 1:
            if type(indexed_pattern[0]) is tuple:
                return [(1,1)]
            else:
                return [1]
        elif get_pattern_size(indexed_pattern) == 2:
            return [1,1]
        elif get_pattern_size(indexed_pattern) == 3:
            return [1,(1,1)]
        elif get_pattern_size(indexed_pattern) == 4:
            return [(1,1),(1,1)]
        else:
            raise err.PatternToDebugError("This dimension 1 pattern is unknown: {}".format(indexed_pattern))

    elif len(indexed_pattern) == 1:
        return [extract_pattern_from_indexed_pattern(indexed_pattern[0])]
    elif len(indexed_pattern) > 2:
        raise err.PatternToDebugError("Nest of more than 2 patterns: {}. Shouldn't happen.".format(indexed_pattern))
    else:
        return [extract_pattern_from_indexed_pattern(indexed_pattern[0]), extract_pattern_from_indexed_pattern(indexed_pattern[1])]

