# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:08:18 2019

@author: amarmore
"""
import math
import music21

import polytopes.data_manipulation as dm
import polytopes.model.errors as err

notes_sharp = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
notes_flat = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']

### Basic chord manipulation
# We only consider minor and major triads
def triad_symbol_from_notes(triad_notes):
    notes = reindex_inversed_triad(triad_notes)
    triad = notes_sharp[notes[0]]
    if (notes[1] - notes[0])%12 == 3:
        triad += "m"
    return triad

def root_from_notes(notes):
    notes = reindex_inversed_triad(notes)
    return notes_sharp[notes[0]]

def triad_notes_from_symbol(symbol, triadic_reduction = True):
    symbol = little_format_symbol(symbol)
    if not is_maj_min_triad_from_symbol(symbol):
        if not triadic_reduction:
            raise err.NotAMajMinTriadException(f"This symbol ({symbol}) does not correspond to a major or minor triad, which is our only case study.")
        else:
            symbol = maj_min_triad_reduction_of_symbol(symbol)
    notes = []
    root_name = root_from_symbol(symbol)
    for i in range(12):
        if notes_sharp[i] == root_name or notes_flat[i] == root_name:
            root = i
            notes.append(root)
            break
    if "m" in symbol:
        notes.append((root+3)%12)
    else:
        notes.append((root+4)%12)
    notes.append((root+7)%12)
    return notes

def root_from_symbol(symbol):
    root_name = symbol[0]
    if len(symbol) > 1:
        if symbol[1] == '-' or symbol[1] == 'b':
            root_name += 'b'
        elif symbol[1] == '+' or symbol[1] == '#':
            root_name += '#'
    return root_name

def reindex_inversed_triad(list_of_notes_numbers):
    if len(list_of_notes_numbers) != 3:
        raise err.NotAMajMinTriadException(f"Too many notes for this chord to be a triad ({list_of_notes_numbers}).")
    for i in range(3):
        root = list_of_notes_numbers[i]
        first_gap = (list_of_notes_numbers[(i+1)%3] - root)%12
        second_gap = (list_of_notes_numbers[(i+2)%3] - root)%12
        if first_gap == 7:
            if second_gap == 3 or second_gap == 4:
                return [root, list_of_notes_numbers[(i+2)%3], list_of_notes_numbers[(i+1)%3]]
        elif second_gap == 7:
            if first_gap == 3 or first_gap == 4:
                return [root, list_of_notes_numbers[(i+1)%3], list_of_notes_numbers[(i+2)%3]]
    raise err.NotAMajMinTriadException(f"These notes ({list_of_notes_numbers}) does not correspond to a major or minor triad, which is our only case study.")

def is_maj_min_triad_from_notes(list_of_notes_numbers):
    if len(list_of_notes_numbers) != 3:
        return False
    for i in range(3):
        root = list_of_notes_numbers[i]
        first_gap = (list_of_notes_numbers[(i+1)%3] - root)%12
        second_gap = (list_of_notes_numbers[(i+2)%3] - root)%12
        if first_gap == 7:
            if second_gap == 3 or second_gap == 4:
                return True
        elif second_gap == 7:
            if first_gap == 3 or first_gap == 4:
                return True
    return False

def is_maj_min_triad_from_symbol(symbol):
    symbol = little_format_symbol(symbol)
    root = root_from_symbol(symbol)
    idx_post_root = len(root)
    return len(symbol) == idx_post_root or (len(symbol) == idx_post_root + 1 and symbol[idx_post_root] == "m")

def maj_min_triad_reduction_of_symbol(symbol, to_sharp = True):
    """
    Reducing a chord to its maj or min triad
    
    Returns
    -------
    string: the triad symbol of the chord.
    """
    symbol = little_format_symbol(symbol)
    root = root_from_symbol(symbol)
    if to_sharp:
        if root not in notes_sharp:
            root_idx = notes_flat.index(root)
            root = notes_sharp[root_idx]
    idx_post_root = len(root)
    if len(symbol) > idx_post_root and symbol[idx_post_root] in ["m","d"]: # minor or diminished triad
        root += "m"
    return root

def little_format_symbol(symbol):
    if "maj" in symbol:
        symbol = symbol.replace("maj","")
    if "min" in symbol:
        symbol = symbol.replace("min","m")
    if "dim" in symbol:
        symbol = symbol.replace("dim","d")
    return symbol
