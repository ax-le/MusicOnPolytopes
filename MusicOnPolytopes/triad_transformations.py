# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:45:22 2021

@author: amarmore
"""

import math
import music21
import itertools

import polytopes.data_manipulation as dm
import polytopes.model.errors as err
import polytopes.model.triad_manipulation as tm
import polytopes.voiceleading_utilities as vl

circle_flat = ['C','Am','F','Dm','Bb','Gm','Eb','Cm','Ab','Fm','Db','Bbm','Gb','Ebm','B','Abm','E','Dbm','A','Gbm','D','Bm','G','Em']
circle_sharp =  ['C','Am','F','Dm','A#','Gm','D#','Cm','G#','Fm','C#','A#m','F#','D#m','B','G#m','E','C#m','A','F#m','D','Bm','G','Em']
chromatic_circle_sharp = ['Cm', 'C', 'C#m', 'C#', 'Dm', 'D', 'D#m', 'D#', 'Em', 'E', 'Fm', 'F', 'F#m', 'F#', 'Gm', 'G', 'G#m', 'G#', 'Am', 'A', 'A#m', 'A#', 'Bm', 'B']
three_five_torus_sharp = ['C','Fm','C#','F#m','D','Gm','D#','G#m','E','Am','F','A#m','F#','Bm','G','Cm','G#','C#m','A','Dm','A#','D#m','B','Em'] 

# %% Voice leading, in optimal transport
# Use Dmitri Tymoczko module instead
def get_voice_leading_distance_symbol(triad_symbol_a, triad_symbol_b, triadic_reduction = True):
    return get_voice_leading_distance_notes(tm.triad_notes_from_symbol(triad_symbol_a, triadic_reduction = triadic_reduction), tm.triad_notes_from_symbol(triad_symbol_b, triadic_reduction = triadic_reduction))

def get_voice_leading_distance_notes(triad_a, triad_b):
    if not tm.is_maj_min_triad_from_notes(triad_a):
        raise err.NotAMajMinTriadException(f"This chord ({triad_a}) does not correspond to a major or minor triad, which is our only case study.")
    if not tm.is_maj_min_triad_from_notes(triad_b):
        raise err.NotAMajMinTriadException(f"This chord ({triad_b}) does not correspond to a major or minor triad, which is our only case study.")
    return vl.nonbijective_vl(triad_a, triad_b)[0]

## Simplistic voice leading
def get_voice_leading_transformation_symbol(triad_symbol_a, triad_symbol_b, triadic_reduction = True):
    return get_voice_leading_transformation_notes(tm.triad_notes_from_symbol(triad_symbol_a, triadic_reduction = triadic_reduction), tm.triad_notes_from_symbol(triad_symbol_b, triadic_reduction = triadic_reduction))

def get_voice_leading_transformation_notes(triad_a, triad_b):
    if not tm.is_maj_min_triad_from_notes(triad_a):
        raise err.NotAMajMinTriadException(f"This chord ({triad_a}) does not correspond to a major or minor triad, which is our only case study.")
    if not tm.is_maj_min_triad_from_notes(triad_b):
        raise err.NotAMajMinTriadException(f"This chord ({triad_b}) does not correspond to a major or minor triad, which is our only case study.")
    return [triad_b[i] - triad_a[i] for i in range(3)]

## Work In Progress
def apply_voice_leading_transformation(triad, transformation):
    #TODO: reflechir à cette fonction
    if not tm.is_maj_min_triad_from_notes(triad):
        raise err.NotAMajMinTriadException(f"This chord ({triad}) does not correspond to a major or minor triad, which is our only case study.")
    if len(transformation) != 3:
        raise NotImplementedError("Not a valid transformation, better error TODO")
    print("Check that it always returns a triad, or act consequently")
    return [(triad[i] + transformation[i])%12 for i in range(3)]

# %% Triadic relation
def get_triadic_position_symbol(triad_symbol, triadic_reduction = True): 
    if not tm.is_maj_min_triad_from_symbol(triad_symbol) and not triadic_reduction:
        raise err.NotAMajMinTriadException(f"This chord ({triad_symbol}) does not correspond to a major or minor triad, which is our only case study.")
    triad = tm.maj_min_triad_reduction_of_symbol(triad_symbol, to_sharp = True)
    return circle_sharp.index(triad)

def get_triadic_position_notes(triad_notes): 
    if not tm.is_maj_min_triad_from_notes(triad_notes):
        raise err.NotAMajMinTriadException(f"This chord ({triad_notes}) does not correspond to a major or minor triad, which is our only case study.")
    symb = tm.triad_symbol_from_notes(triad_notes)
    return get_triadic_position_symbol(symb)

    
def triadic_mvt_triads(first_triad_symbol, second_triad_symbol, triadic_reduction = True):
    """
    Compute the movement/relation between both chords in the circle of triads.
    
    It depends on the circle, and can be either on circle of fifth (chromatic) or on the 3-5 Torus (!chromatic)
    NB: Now, only the circle of fifth is tolerated. Could be temporary or permanent.

    Parameters
    ----------
    first_chord_symbol, second_chord_symbol : strings
        The symbols of both chords which relation is to compute.
    
    Returns
    -------
    mvt : integer
        The relation between both chords.

    """
    mvt = (get_triadic_position_symbol(second_triad_symbol, triadic_reduction = triadic_reduction) - get_triadic_position_symbol(first_triad_symbol, triadic_reduction = triadic_reduction))%24
    if mvt > 12:
        mvt -= 24
    return mvt

def apply_triadic_mvt(triad_symbol, mvt, triadic_reduction = True):
    """
    Apply a movement/relation to a chord in the circle of triads.

    Parameters
    ----------
    triad_symbol : string
        The symbol of the triad, on which to aply the movement.
    mvt : integer
        The relation to apply.
        
    Returns
    -------
    string
        The new triad symbol, after applying the relation.

    """
    if not tm.is_maj_min_triad_from_symbol(triad_symbol) and not triadic_reduction:
        raise err.NotAMajMinTriadException(f"This chord ({triad_symbol}) does not correspond to a major or minor triad, which is our only case study.")
    triad = tm.maj_min_triad_reduction_of_symbol(triad_symbol, to_sharp = True)
    idx = get_triadic_position_symbol(triad)
    return circle_sharp[(idx + mvt)%24]


# %% Triadic chromatic relation
def get_chromatic_position_symbol(triad_symbol, triadic_reduction = True): 
    if not tm.is_maj_min_triad_from_symbol(triad_symbol) and not triadic_reduction:
        raise err.NotAMajMinTriadException(f"This chord ({triad_symbol}) does not correspond to a major or minor triad, which is our only case study.")
    triad = tm.maj_min_triad_reduction_of_symbol(triad_symbol, to_sharp = True)
    return chromatic_circle_sharp.index(triad)

def get_chromatic_position_notes(triad_notes): 
    if not tm.is_maj_min_triad_from_notes(triad_notes):
        raise err.NotAMajMinTriadException(f"This chord ({triad_notes}) does not correspond to a major or minor triad, which is our only case study.")
    symb = tm.triad_symbol_from_notes(triad_notes)
    return get_chromatic_position_symbol(symb)
    
def chromatic_mvt_triads(first_triad_symbol, second_triad_symbol, triadic_reduction = True):
    mvt = (get_chromatic_position_symbol(second_triad_symbol, triadic_reduction = triadic_reduction) - get_chromatic_position_symbol(first_triad_symbol, triadic_reduction = triadic_reduction))%24
    if mvt > 12:
        mvt -= 24
    return mvt

def apply_chromatic_mvt(triad_symbol, mvt, triadic_reduction = True):
    if not tm.is_maj_min_triad_from_symbol(triad_symbol) and not triadic_reduction:
        raise err.NotAMajMinTriadException(f"This chord ({triad_symbol}) does not correspond to a major or minor triad, which is our only case study.")
    triad = tm.maj_min_triad_reduction_of_symbol(triad_symbol, to_sharp = True)
    idx = get_chromatic_position_symbol(triad)
    return chromatic_circle_sharp[(idx + mvt)%24]

# %% Triadic 3-5 torus relation
def get_three_five_torus_position_symbol(triad_symbol, triadic_reduction = True): 
    if not tm.is_maj_min_triad_from_symbol(triad_symbol) and not triadic_reduction:
        raise err.NotAMajMinTriadException(f"This chord ({triad_symbol}) does not correspond to a major or minor triad, which is our only case study.")
    triad = tm.maj_min_triad_reduction_of_symbol(triad_symbol, to_sharp = True)
    return three_five_torus_sharp.index(triad)

def get_three_five_torus_position_notes(triad_notes): 
    if not tm.is_maj_min_triad_from_notes(triad_notes):
        raise err.NotAMajMinTriadException(f"This chord ({triad_notes}) does not correspond to a major or minor triad, which is our only case study.")
    symb = tm.triad_symbol_from_notes(triad_notes)
    return get_three_five_torus_position_symbol(symb)
    
def three_five_torus_mvt_triads(first_triad_symbol, second_triad_symbol, triadic_reduction = True):
    mvt = (get_three_five_torus_position_symbol(second_triad_symbol, triadic_reduction = triadic_reduction) - get_three_five_torus_position_symbol(first_triad_symbol, triadic_reduction = triadic_reduction))%24
    if mvt > 12:
        mvt -= 24
    return mvt

def apply_three_five_torus_mvt(triad_symbol, mvt, triadic_reduction = True):
    if not tm.is_maj_min_triad_from_symbol(triad_symbol) and not triadic_reduction:
        raise err.NotAMajMinTriadException(f"This chord ({triad_symbol}) does not correspond to a major or minor triad, which is our only case study.")
    triad = tm.maj_min_triad_reduction_of_symbol(triad_symbol, to_sharp = True)
    idx = get_three_five_torus_position_symbol(triad)
    return three_five_torus_sharp[(idx + mvt)%24]
   
# %% Tonnetz distances, for triads
def triadic_tonnetz_relation_notes(triad_notes_a, triad_notes_b):
    a = music21.chord.Chord(triad_notes_a)
    c_1 = music21.analysis.neoRiemannian._simplerEnharmonics(a)
    
    b = music21.chord.Chord(triad_notes_b)
    c_2 = music21.analysis.neoRiemannian._simplerEnharmonics(b)
    
    if c_2.orderedPitchClasses == c_1.orderedPitchClasses:
        return 0
    
    for i in range(1,9):
        for permut in itertools.product('LPR', repeat=i):
            try:
                c_1_permuted = music21.analysis.neoRiemannian.LRP_combinations(c_1, permut, raiseException=True, simplifyEnharmonics=True)
            except music21.analysis.neoRiemannian.LRPException:
                # Happens sometimes for now, but should be solved in near future
                c_1_permuted = manually_loop_lrp_comb(c_1, permut)

            if c_2.orderedPitchClasses == c_1_permuted.orderedPitchClasses:
                return permut
    raise NotImplementedError("Not found, try increasing the accepted amout of relations.")
    
def triadic_tonnetz_distance_notes(triad_notes_a, triad_notes_b):
    relation = triadic_tonnetz_relation_notes(triad_notes_a, triad_notes_b)
    if relation == 0 or relation == None:
        return 0
    return len(relation)
    
def triadic_tonnetz_relation_symbol(triad_symbol_a, triad_symbol_b):
    return triadic_tonnetz_relation_notes(tm.triad_notes_from_symbol(triad_symbol_a), tm.triad_notes_from_symbol(triad_symbol_b))

def triadic_tonnetz_distance_symbol(triad_symbol_a, triad_symbol_b):
    relation = triadic_tonnetz_relation_symbol(triad_symbol_a, triad_symbol_b)
    if relation == 0 or relation == None:
        return 0
    return len(relation)

def manually_loop_lrp_comb(chord, relations):
    while len(relations) > 0:
        if relations[-1] == "P":
            chord = music21.analysis.neoRiemannian.P(chord)
        elif relations[-1] == "L":
            chord = music21.analysis.neoRiemannian.L(chord)
        elif relations[-1] == "R":
            chord = music21.analysis.neoRiemannian.R(chord)
        else:
            raise err.InvalidArgumentValueException(f"Invalid relation type: {relations[-1]}, only L, P and R are accepted.")
        chord = music21.analysis.neoRiemannian._simplerEnharmonics(chord)
        relations = relations[:-1]
    return chord

def apply_triadic_tonnetz_relation_notes(triad_notes_a, LRP_relation):
    a = music21.chord.Chord(triad_notes_a)
    c_1 = music21.analysis.neoRiemannian._simplerEnharmonics(a)
    
    if LRP_relation == 0 or LRP_relation == None:
        return triad_notes_a
    
    try:
        c_transformed = music21.analysis.neoRiemannian.LRP_combinations(c_1, LRP_relation, raiseException=True, simplifyEnharmonics=True)
    except music21.analysis.neoRiemannian.LRPException:
        # Happens sometimes for now, but should be solved in near future
        c_transformed = manually_loop_lrp_comb(c_1, LRP_relation)
    notes = []
    for a_pitch in c_transformed.pitches:
        notes.append(int(a_pitch.ps)%12)
    return tm.reindex_inversed_triad(notes)

def apply_triadic_tonnetz_relation_symbol(triad_symbol_a, LRP_relation):
    if LRP_relation == 0:
        return triad_symbol_a
    notes = tm.triad_notes_from_symbol(triad_symbol_a)
    notes_transformed = apply_triadic_tonnetz_relation_notes(notes, LRP_relation)
    return tm.triad_symbol_from_notes(notes_transformed)
    
