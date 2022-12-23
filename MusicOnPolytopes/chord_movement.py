# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:08:18 2019

@author: amarmore
"""
import math

import polytopes.data_manipulation as dm
from polytopes.model.note import Note
from polytopes.model.chord import Chord
import polytopes.model.errors as err
from polytopes.model.constants import Constants as cst

# %% Voice leading, in optimal transport
# def ot_notes(first_note_number, second_note_number, fifth=False):
#     """
#     TODO
#     """
#     if fifth:
#         circle_of_fifth = shared.get_circle_of_fifth()
#         mvt = (circle_of_fifth[second_note_number] - circle_of_fifth[first_note_number])%12
#     else:
#         mvt = (second_note_number - first_note_number)%12
#     if mvt > 6: # Relative movement
#         mvt = mvt-12
#     return mvt

# def ot_chords(first_chord_as_list, second_chord_as_list, fifth=False):
#     """
#     TODO
#     """
#     if len(first_chord_as_list) == len(second_chord_as_list):
#         return [ot_notes(first_chord_as_list[i], second_chord_as_list[i], fifth=fifth) for i in range(len(first_chord_as_list))]
#     else:
#         raise err.InvalidChordException("Both chords are not of the same size in the calculation of the voice leading.") from None
           
# def apply_vl_ot_chords(a_chord, relation, fifth=False):
#     """
#     """
#     displaced_chord_notes = []
#     if len(a_chord) != len(relation):
#         raise err.InvalidChordException('The chord and the relation have different size, cannot be applied.') from None
    
#     for idx in range(len(a_chord)):
#         if fifth:
#             circle_of_fifth = shared.get_circle_of_fifth()
#             reversed_circle = shared.get_reversed_circle_of_fifth()
            
#             idx_in_circle = (circle_of_fifth[a_chord[idx]] + relation[idx])%12
#             displaced_chord_notes.append(Note(reversed_circle[idx_in_circle]))
#         else:
#             displaced_chord_notes.append(Note((a_chord[idx] + relation[idx])%12))
#     return displaced_chord_notes

# %% Triadic relation
def get_triadic_position(chord, chromatic = True):
    """
    A function that computes the position of the chord in the triadic circle.
    
    Parameters
    ----------
    chord: Chord object, or symbol or list of integers
        The chord to find in the circle of triads
    chromatic: boolean
        Whether to return the circle of triads in the chromatic order or in the 3-5 torus order
        
    Returns
    -------
    integer: the index position of the chord in the circle of triads.
    """
    circle_flat = dm.get_circle_of_triads(flat = True, chromatic = chromatic)
    circle_sharp = dm.get_circle_of_triads(flat = False, chromatic = chromatic)
    if dm.is_a_chord_object(chord):
        if chord.triad != cst.AMBIGUOUS:
            chord = chord.triad
            symb = sharp_to_flat(chord)
            return circle_flat.index(symb)
        else:
            raise err.InvalidChordException('Chord object, but without triad {}.'.format(chord.symbol)) from None
    elif chord not in circle_flat and chord not in circle_sharp:
        symbol = chord[0]
        idx_parsing = 1
        if chord[idx_parsing] in ["b", "#"]:
            symbol += chord[idx_parsing]
            if chord[idx_parsing] =="#":
                symbol = sharp_to_flat(chord[:idx_parsing+1])
            idx_parsing += 1
        
        if len(chord) > idx_parsing and chord[idx_parsing] == "m": 
            if len(chord) == idx_parsing + 1 or chord[idx_parsing + 1] == "i": # Minor triad
                symbol += "m"
        return circle_flat.index(symbol)
    elif chord in circle_flat:
        return circle_flat.index(chord)
    elif  chord in circle_sharp:
        return circle_sharp.index(chord)
    else:
        raise NotImplementedError("What case is left?")

    
def sharp_to_flat(symbol):
    """
    Converting the symbol from the sharp to the flat convention.

    Parameters
    ----------
    symbol : string
        The symbol of the chord.

    Returns
    -------
    string
        The symbol of the chord, in the flat convention.

    """
    index = get_notes_sharp().index(symbol)
    return get_notes_flat()[index]
    
def get_notes_sharp():
    """
    All the symbols of notes in the sharp convention.
    
    Returns
    -------
    List of the symbols.
    """
    return ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                
def get_notes_flat():
    """
    All the symbols of notes in the flat convention.
    
    Returns
    -------
    List of the symbols.
    """
    return ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
    
def triadic_mvt_chords(first_chord_symbol, second_chord_symbol, chromatic = True):
    """
    Compute the movement/relation between both chords in the circle of triads.
    
    It depends on the circle, and can be either on circle of fifth (chromatic) or on the 3-5 Torus (!chromatic)
    NB: Now, only the circle of fifth is tolerated. Could be temporary or permanent.

    Parameters
    ----------
    first_chord_symbol, second_chord_symbol : strings
        The symbols of both chords which relation is to compute.
    chromatic : bollean, optional
        Choice of the circle of triads, either circle of fifth (chromatic = True) or on the 3-5 Torus (chromatic = False).
        The default is True. (In fact, only the circle of fifth exists now)

    Returns
    -------
    mvt : integer
        The relation between both chords.

    """
    mvt = (get_triadic_position(second_chord_symbol, chromatic = chromatic) - get_triadic_position(first_chord_symbol, chromatic = chromatic))%24
    if mvt > 12:
        mvt -= 24
    return mvt

def apply_triadic_mvt(chord, mvt, chromatic = True):
    """
    Apply a movement/relation to a chord in the circle of triads.

    Parameters
    ----------
    chord : string
        The chord, on which to aply the movement.
    mvt : integer
        The relation to apply.
    chromatic : bollean, optional
        Choice of the circle of triads, either circle of fifth (chromatic = True) or on the 3-5 Torus (chromatic = False).
        The default is True. (In fact, only the circle of fifth exists now)
        
    Returns
    -------
    string
        The new chord, after applying the relation.

    """
    circle = dm.get_circle_of_triads(flat = False, chromatic = chromatic)
    idx = get_triadic_position(chord, chromatic = chromatic)
    return circle[(idx + mvt)%24]

# %% Movement norms
def score_note_leading(mvt):
    """
    Absolute value of the movement.
    
    Parameters
    ----------
    mvt: integer
        The movement to measure
        
    Returns
    -------
    integer: Absolute value of the movement
    """
    return abs(mvt)

def l1_norm(voice_leading):
    """
    Sum of the absolute value of the movements (l_1 norm).
    
    Parameters
    ----------
    voice_leading: list of integer
        The movement to measure
        
    Returns
    -------
    integer: l1 norm of the movement
    """
    try: # If it's an integer, it's a triadic displacement. # TODO: not really a norm, and not extended to the others (as it's useless)
        triadic_movement = int(voice_leading)
        return score_note_leading(triadic_movement)
    except (ValueError, TypeError): # Else, if it's a tab, it's an optimal transport.
        s = 0
        for mvt in voice_leading:
            s += score_note_leading(mvt)
        return s

def l2_norm(voice_leading):
    """
    Euclidian distance between two chords (or l_2 norm of the relation): square root of the sum of the squared value of the movements between pairs of notes.
    
    Parameters
    ----------
    voice_leading: list of integer
        The movement to measure
        
    Returns
    -------
    integer: l2 norm of the movement
    """
    try: # If it's an integer, it's a triadic displacement. # TODO: not really a norm, and not extended to the others (as it's useless)
        triadic_movement = int(voice_leading)
        return score_note_leading(triadic_movement)
    except (ValueError, TypeError): # Else, if it's a tab, it's an optimal transport.
        s = 0
        for mvt in voice_leading:
            a = score_note_leading(mvt)
            s += a*a
        return math.sqrt(s)

def linf_norm(voice_leading):
    """
    Maximal absolute value of the movements (infinite norm).
    
    Parameters
    ----------
    voice_leading: list of integer
        The movement to measure
        
    Returns
    -------
    integer: l_infinite norm of the movement
    """
    try: # If it's an integer, it's a triadic displacement. # TODO: not really a norm, and not extended to the others (as it's useless)
        triadic_movement = int(voice_leading)
        return score_note_leading(triadic_movement)
    except (ValueError, TypeError): # Else, if it's a tab, it's an optimal transport.
        s = 0
        for mvt in voice_leading:
            a = score_note_leading(mvt)
            if a > s:
                s = a
        return s
