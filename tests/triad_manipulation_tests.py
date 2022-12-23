# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:38:53 2019

@author: amarmore
"""

import unittest
import polytopes.data_manipulation as dm
import polytopes.model.errors as err
import polytopes.model.triad_manipulation as tm

import numpy as np

class TriadManipulationTests(unittest.TestCase):

##################################### Starting from the notes ################################
        
    def test_symbol_from_notes(self):
        """
        Tests if the symbol is correctly retrieved from the notes.
        """
        chord_maj = [8, 0, 3]
        chord_min = [8, 11, 3]
        chord_sus4 = [8, 1, 3]
        chord_sus2 = [8, 10, 3]
        chord_aug = [8, 0, 4]
        chord_dim = [8, 0, 2]
        
        self.assertEqual(tm.triad_symbol_from_notes(chord_maj), "G#")
        self.assertEqual(tm.triad_symbol_from_notes(chord_min), "G#m")
        with self.assertRaises(err.NotAMajMinTriadException):
            tm.triad_symbol_from_notes(chord_sus4)
        with self.assertRaises(err.NotAMajMinTriadException):
            tm.triad_symbol_from_notes(chord_sus2)
        with self.assertRaises(err.NotAMajMinTriadException):
            tm.triad_symbol_from_notes(chord_aug)
        with self.assertRaises(err.NotAMajMinTriadException):
            tm.triad_symbol_from_notes(chord_dim)

        
    def test_root_from_notes(self):
        """
        Tests if the root is correctly retrieved from the notes.
        """
        chord = [8, 11, 3]
        self.assertTrue(tm.root_from_notes(chord), "G#")

        chord = [3, 11, 8]
        self.assertTrue(tm.root_from_notes(chord), "G#")
        
        chord = [8, 1, 3]
        with self.assertRaises(err.NotAMajMinTriadException):
            tm.root_from_notes(chord)


##################################### Starting from the symbol ################################

    def test_notes_from_symbol(self):
        """
        Tests if the notes are correctly retrieved from the symbol.
        """
        notes = [8, 11, 3]
        self.assertEqual(tm.triad_notes_from_symbol("Abm"), notes)
        self.assertEqual(tm.triad_notes_from_symbol("Abmin"), notes)
        self.assertEqual(tm.triad_notes_from_symbol("G#m"), notes)
        self.assertEqual(tm.triad_notes_from_symbol("G#min"), notes)
        self.assertEqual(tm.triad_notes_from_symbol("Abd", triadic_reduction = True), notes)
        self.assertEqual(tm.triad_notes_from_symbol("Abdim", triadic_reduction = True), notes)
        with self.assertRaises(err.NotAMajMinTriadException):
            self.assertEqual(tm.triad_notes_from_symbol("Abdim", triadic_reduction = False), notes) 

        notes = [8, 0, 3]
        self.assertEqual(tm.triad_notes_from_symbol("Ab"), notes)
        self.assertEqual(tm.triad_notes_from_symbol("Abmaj"), notes)
        self.assertEqual(tm.triad_notes_from_symbol("G#"), notes)
        self.assertEqual(tm.triad_notes_from_symbol("G#maj"), notes)
        
        self.assertEqual(tm.triad_notes_from_symbol("G#5", triadic_reduction = True), notes)  
        with self.assertRaises(err.NotAMajMinTriadException):
            self.assertEqual(tm.triad_notes_from_symbol("G#5", triadic_reduction = False), notes)  
            
        self.assertEqual(tm.triad_notes_from_symbol("Absus4", triadic_reduction = True), notes)  
        with self.assertRaises(err.NotAMajMinTriadException):
            self.assertEqual(tm.triad_notes_from_symbol("Absus4", triadic_reduction = False), notes)  
            
        self.assertEqual(tm.triad_notes_from_symbol("G#7s4", triadic_reduction = True), notes)  
        with self.assertRaises(err.NotAMajMinTriadException):
            self.assertEqual(tm.triad_notes_from_symbol("G#7s4", triadic_reduction = False), notes)  

    def test_root_from_symbol(self):
        """
        Tests if the root is correctly retrieved from the symbol.
        """
        self.assertTrue(tm.root_from_symbol('Asus2') == 'A')
        self.assertTrue(tm.root_from_symbol('D') == 'D')
        self.assertTrue(tm.root_from_symbol('Db') == 'Db')
        self.assertTrue(tm.root_from_symbol('Dbm') == 'Db')
        self.assertTrue(tm.root_from_symbol('Cm') == 'C')
        self.assertTrue(tm.root_from_symbol('C#') == 'C#')
        self.assertTrue(tm.root_from_symbol('C#m') == 'C#')
        self.assertTrue(tm.root_from_symbol('C#dim') == 'C#')
        self.assertTrue(tm.root_from_symbol('Abmin') == 'Ab')
        
    def test_little_format_symbol(self):
        self.assertTrue(tm.little_format_symbol('C#dim') == 'C#d')
        self.assertTrue(tm.little_format_symbol('C#min') == 'C#m')
        self.assertTrue(tm.little_format_symbol('C#maj') == 'C#')


    def test_reindex_inversed_triad(self):
        chord = [8, 11, 3]
        self.assertEqual(tm.reindex_inversed_triad([8, 11, 3]), chord)
        self.assertEqual(tm.reindex_inversed_triad([8, 3, 11]), chord)
        self.assertEqual(tm.reindex_inversed_triad([3, 11, 8]), chord)
        self.assertEqual(tm.reindex_inversed_triad([3, 8, 11]), chord)
        self.assertEqual(tm.reindex_inversed_triad([11, 8, 3]), chord)
        self.assertEqual(tm.reindex_inversed_triad([11, 3, 8]), chord)
        
        chord_maj = [8, 0, 3]
        self.assertEqual(tm.reindex_inversed_triad([0, 3, 8]), chord_maj)
        self.assertEqual(tm.reindex_inversed_triad([0, 8, 3]), chord_maj)

    def test_is_maj_min_triad_from_notes(self):
        self.assertTrue(tm.is_maj_min_triad_from_notes([8, 11, 3]))
        self.assertTrue(tm.is_maj_min_triad_from_notes([8, 0, 3]))
        self.assertFalse(tm.is_maj_min_triad_from_notes([8, 1, 3]))

    def test_is_maj_min_triad_from_symbol(self):
        self.assertTrue(tm.is_maj_min_triad_from_symbol("Abm"))
        self.assertTrue(tm.is_maj_min_triad_from_symbol("G#"))
        self.assertFalse(tm.is_maj_min_triad_from_symbol("Abd"))
        
    def test_is_maj_min_triad_from_symbol(self):
        self.assertEqual(tm.maj_min_triad_reduction_of_symbol("Abm"), "G#m")
        self.assertEqual(tm.maj_min_triad_reduction_of_symbol("G#"), "G#")
        self.assertEqual(tm.maj_min_triad_reduction_of_symbol("Abd"), "G#m")
        self.assertEqual(tm.maj_min_triad_reduction_of_symbol("G#sus2"), "G#")


if __name__ == '__main__':
    unittest.main()