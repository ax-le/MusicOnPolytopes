# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:38:53 2019

@author: amarmore
"""

import unittest
import polytopes.model.triad_manipulation as tm
import polytopes.triad_transformations as tt
import polytopes.model.errors as err
import polytopes.chord_movement as mvt

import numpy as np

class TriadTransformationsTests(unittest.TestCase):
            
    def test_get_voice_leading_distance_symbol(self):
        self.assertEqual(tt.get_voice_leading_distance_symbol("F","Am"), 1)
        self.assertEqual(tt.get_voice_leading_distance_symbol("Fmaj","Amin"), 1)
        self.assertEqual(tt.get_voice_leading_distance_symbol("F","Amin"), 1)
        self.assertEqual(tt.get_voice_leading_distance_symbol("Fmaj","Am"), 1)
        
        self.assertEqual(tt.get_voice_leading_distance_symbol("Dm","Am"), 3)
        self.assertEqual(tt.get_voice_leading_distance_symbol("Bb","F#m"), 3)
        self.assertEqual(tt.get_voice_leading_distance_symbol("C#","F#m"), 2)
        self.assertEqual(tt.get_voice_leading_distance_symbol("C#","F#dim", triadic_reduction = True), 2)
        with self.assertRaises(err.NotAMajMinTriadException):
            tt.get_voice_leading_distance_symbol("Cdim","Asus2", triadic_reduction = False)
        
    def test_get_voice_leading_distance_notes(self):
        self.assertEqual(tt.get_voice_leading_distance_notes([8,0,3],[8,11,3]), 1)
        self.assertEqual(tt.get_voice_leading_distance_notes([11,3,6],[8,11,3]), 2)
        with self.assertRaises(err.NotAMajMinTriadException):
            tt.get_voice_leading_distance_notes([11,3,6],[8,1,3])
                                                    
    def test_get_voice_leading_tranformation_notes(self):
        self.assertEqual(tt.get_voice_leading_transformation_notes([8,0,3],[8,11,3]), [0,11,0])
        self.assertEqual(tt.get_voice_leading_transformation_notes([11,3,6],[8,11,3]), [-3, 8, -3])
        with self.assertRaises(err.NotAMajMinTriadException):
            tt.get_voice_leading_transformation_notes([11,3,6],[8,1,3])
            
    def test_get_voice_leading_tranformation_symbol(self):
        self.assertEqual(tt.get_voice_leading_transformation_symbol("Ab","Abm"), [0,11,0])
        self.assertEqual(tt.get_voice_leading_transformation_symbol("B","Abm"), [-3, 8, -3])
        self.assertEqual(tt.get_voice_leading_transformation_symbol("B","Absus2"), [-3, -3, -3])

        with self.assertRaises(err.NotAMajMinTriadException):
            tt.get_voice_leading_transformation_symbol("A","Asus2", triadic_reduction = False)
            
    def test_get_triadic_position_symbol(self):
        self.assertEqual(tt.get_triadic_position_symbol("Am"), 1)
        self.assertEqual(tt.get_triadic_position_symbol("Gbm"), 19)
        self.assertEqual(tt.get_triadic_position_symbol("F#m"), 19)
        self.assertEqual(tt.get_triadic_position_symbol("F#dim", triadic_reduction=True), 19)
        with self.assertRaises(err.NotAMajMinTriadException):
            tt.get_triadic_position_symbol("F#dim", triadic_reduction=False)
            
    def test_get_triadic_position_notes(self):
        self.assertEqual(tt.get_triadic_position_notes([9,0,4]), 1)
        self.assertEqual(tt.get_triadic_position_notes([8,0,3]), 8)
        with self.assertRaises(err.NotAMajMinTriadException):
            tt.get_triadic_position_notes([8,1,3])

    def test_triadic_mvt_triads(self):
        self.assertEqual(tt.triadic_mvt_triads("C","Em"), -1)
        self.assertEqual(tt.triadic_mvt_triads("Abm","A"), 3)
        self.assertEqual(tt.triadic_mvt_triads("C#","Gbm"), 9)
        self.assertEqual(tt.triadic_mvt_triads("C#","Gbdim", triadic_reduction=True), 9)
        with self.assertRaises(err.NotAMajMinTriadException):
            tt.triadic_mvt_triads("C#","Gbdim", triadic_reduction=False)
            
    def test_triad_and_mvt(self):
        circle_sharp =  ['C','Am','F','Dm','A#','Gm','D#','Cm','G#','Fm','C#','A#m','F#','D#m','B','G#m','E','C#m','A','F#m','D','Bm','G','Em']
        for chord_1 in circle_sharp:
            for chord_2 in circle_sharp:
                self.assertEqual(tt.triadic_mvt_triads(chord_1, chord_2), mvt.triadic_mvt_chords(chord_1, chord_2))
            for rel in range(0,24):
                self.assertEqual(tt.apply_triadic_mvt(chord_1, rel), mvt.apply_triadic_mvt(chord_1, rel))
    
    def test_apply_triadic_mvt(self):
        self.assertEqual(tt.apply_triadic_mvt("C", -1), "Em")
        self.assertEqual(tt.apply_triadic_mvt("Abm", 3), "A")
        self.assertEqual(tt.apply_triadic_mvt("D#", 5), "A#m")
        self.assertEqual(tt.apply_triadic_mvt("D#sus2", 5, triadic_reduction = True), "A#m")
        with self.assertRaises(err.NotAMajMinTriadException):
            tt.apply_triadic_mvt("D#sus2", 5, triadic_reduction = False)

    def test_triadic_tonnetz_relation_symbol(self):
        self.assertEqual(tt.triadic_tonnetz_relation_symbol("D","Gm"), ('P', 'L', 'R'))
        self.assertEqual(tt.triadic_tonnetz_relation_symbol("Dm","Db"), ('L', 'P', 'R'))
        self.assertEqual(tt.triadic_tonnetz_relation_symbol("C","A"), ('R', 'P'))
        self.assertEqual(tt.triadic_tonnetz_relation_symbol("C","Db"), ('L','P','R', 'P'))
        
    def test_triadic_tonnetz_distance_symbol(self):
        self.assertEqual(tt.triadic_tonnetz_distance_symbol("D","Gm"), 3)
        self.assertEqual(tt.triadic_tonnetz_distance_symbol("Dm","Db"), 3)
        self.assertEqual(tt.triadic_tonnetz_distance_symbol("C","A"), 2)
        self.assertEqual(tt.triadic_tonnetz_distance_symbol("C","Db"), 4)
        
    def test_triadic_tonnetz_relation_notes(self):
        self.assertEqual(tt.triadic_tonnetz_relation_notes([2,6,9],[7,10,2]), ('P', 'L', 'R'))
        
    def test_triadic_tonnetz_distance_notes(self):
        self.assertEqual(tt.triadic_tonnetz_distance_notes([2,6,9],[7,10,2]), 3)
        
    def test_apply_triadic_tonnetz_relation_symbol(self):
        self.assertEqual(tt.apply_triadic_tonnetz_relation_symbol("D",('P', 'L', 'R')), "Gm")
        self.assertEqual(tt.apply_triadic_tonnetz_relation_symbol("D",('L', 'P', 'R')), "D#m")

    def test_apply_triadic_tonnetz_relation_notes(self):
        self.assertEqual(tt.apply_triadic_tonnetz_relation_notes([2,6,9],('P', 'L', 'R')), [7,10,2])

    def test_chromatic_equals_fifth(self):
        circle_sharp =  ['C','Am','F','Dm','A#','Gm','D#','Cm','G#','Fm','C#','A#m','F#','D#m','B','G#m','E','C#m','A','F#m','D','Bm','G','Em']
        for i in range(len(circle_sharp)):
            chord_1 = circle_sharp[i]
            for j in range(len(circle_sharp)):
                chord_2 = circle_sharp[j]
                rel_triadic = tt.triadic_mvt_triads(chord_1, chord_2)
                triadic = tt.apply_triadic_mvt(chord_1, rel_triadic)
                rel_chromatic = tt.chromatic_mvt_triads(chord_1, chord_2)
                chromatic = tt.apply_chromatic_mvt(chord_1, rel_chromatic)
                self.assertEqual(chromatic, triadic)        

if __name__ == '__main__':
    unittest.main()