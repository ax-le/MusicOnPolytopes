#! /usr/bin/python
# -*- coding: utf-8 -*-

# Encapsulation of the measures of voice leading.

import math

def score_note_leading(mvt):
    """
    Absolute value of the movement.
    """
    return abs(mvt)

def l1_norm(voice_leading):
    """
    Sum of the absolute value of the movements (l_1 norm).
    """
    s = 0
    for mvt in voice_leading:
        s += score_note_leading(mvt)
    return s

def l2_norm(voice_leading):
    """
    Euclidian distance between two chords (or l_2 norm of the relation):
        square root of the sum of the squared value of the movements between pairs of notes.
    """
    s = 0
    for mvt in voice_leading:
        a = score_note_leading(mvt)
        s += a*a
    return math.sqrt(s)

def linf_norm(voice_leading):
    """
    Maximal absolute value of the movements (infinite norm)
    """
    s = 0
    for mvt in voice_leading:
        a = score_note_leading(mvt)
        if a > s:
            s = a
    return s