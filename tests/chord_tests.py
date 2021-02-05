# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:38:53 2019

@author: amarmore
"""

import unittest
import polytopes.data_manipulation as dm
from polytopes.model.note import Note
from polytopes.model.chord import Chord
import polytopes.model.errors as err

import numpy as np
from polytopes.model.constants import Constants as cst

class ChordTests(unittest.TestCase):
            
    def test_is_a_chord_object(self):
        self.assertTrue(dm.is_a_chord_object(Chord([1, 5, 8])))
        self.assertFalse(dm.is_a_chord_object([1, 5, 8]))
        self.assertFalse(dm.is_a_chord_object(Note(1)))
        self.assertTrue(dm.is_a_chord_object(Chord([1, 5, 8, 1], redundancy = True)))
    
    def test_chord_not_a_note(self):
        self.assertFalse(dm.is_a_note_object(Chord([1, 5, 8])))
        
    def test_incorrect_blank_chord(self):
        """
        Checks that instanciating a blank Chord is incorrect.
        """
        with self.assertRaises(err.InvalidChordNotesException):
            Chord([])

    def test_is_note_in_chord(self):
        """
        Checks if the function "Note.is_in_chord(chord)" is correct, meaning that a note is in the tested chord.
        Also checks if the override of __contains__ function works, meaning that the note is in the chord.
        """
        note = Note(1)
        chord_true = Chord([1, 4, 8])
        self.assertTrue(note.is_in_chord(chord_true))
        self.assertTrue(note in chord_true)
        chord_false = Chord([2, 5, 9])
        self.assertFalse(note.is_in_chord(chord_false))
        self.assertFalse(note in chord_false)

    def test_is_redundant(self):
        """
        Checks if redundant notes are correctly added to a Chord with redundancy.
        """
        self.assertEqual(Chord([1, 5, 8, 5], redundancy = True).get_numbers(), [1, 5, 8, 5])
        self.assertTrue(Chord([1, 5, 8, 5], redundancy = True).get_numbers() == [1, 5, 8, 5])
        
    def test_is_note_in_list_of_notes(self):
        """
        Checks if the function "Note.is_in_list_of_notes(list_of_notes)" is correct, meaning that the current note is in this list.
        This function is used to create a chord.
        """
        note = Note(1)
        list_of_notes_true = [Note(1), Note(4), Note(8)]
        self.assertTrue(note.is_in_list_of_notes(list_of_notes_true))
        list_of_notes_false = [Note(2), Note(5), Note(9)]
        self.assertFalse(note.is_in_list_of_notes(list_of_notes_false))

    def test_correct_notes(self):
        """
        Checks if the notes contained in a Chord object are in the correct format.
        """
        notes = [Note(8), Note(11), Note(3)]
        chord = Chord(notes)
        self.assertEqual(notes, chord.notes)
        
        chord = Chord([8, 11, 3])
        self.assertTrue(Note(8) == chord[0])
        self.assertTrue(Note(11) == chord[1])
        self.assertTrue(Note(3) == chord[2])
        self.assertFalse(Note(8) == chord[1])
    
    def test_automatic_traidic_ordering(self):
        """
        Checks that the notes are ordered in the root-third-fifth order when the chord is a triad.
        """
        chord = Chord([3, 11, 8])
        self.assertTrue(Note(8) == chord[0])
        self.assertTrue(Note(11) == chord[1])
        self.assertTrue(Note(3) == chord[2])
        self.assertFalse(Note(8) == chord[1])
                
    def test_eq_override(self):
        """
        Test of the function above.
        """
        basis_chord = Chord([8, 11, 3])
        same_chord = Chord('Abmin')
        self.assertTrue(basis_chord == same_chord)

        added_note = Chord([8, 11, 3, 7])
        self.assertFalse(basis_chord == added_note)

        diff_chord = Chord([8, 0, 3])
        self.assertFalse(basis_chord == diff_chord)
        
        self.assertTrue(basis_chord == Chord([8, 11, 3, 11]))
        
        self.assertFalse(Chord([8, 11, 3, 11], redundancy = False) == Chord([8, 11, 3, 11], redundancy = True))


    def test_get_chord_numbers(self):
        """
        Test of the "get_numbers()" method.
        """
        chord = Chord([8, 11, 3])
        self.assertEqual([8, 11, 3], chord.get_numbers())
    
    def test_set_notes(self):
        """
        Test if the setter for the notes is correct.
        """
        chord = Chord([8, 11, 3])
        self.assertEqual(chord.symbol, 'Abm')
        
        chord.notes = [11, 3, 6]
        self.assertTrue(chord == Chord([11, 3, 6]))
        self.assertTrue(chord == Chord([11, 3, 6, 11]))
        self.assertFalse(chord == Chord([8, 11, 3]))

        self.assertEqual(chord.symbol, 'B')
        self.assertNotEqual(chord.symbol, 'Abm')

        self.assertEqual(chord.root, Note('B'))
        self.assertEqual(chord.triad, 'B')

    def test_cannot_wrongly_set_notes(self):
        """
        Checks that wrongly formatted notes can't be set by the constructor or the setter.
        """
        with self.assertRaises(err.InvalidChordNotesException):
            chord = Chord([13, 3, 6])
        chord = Chord([8, 11, 3])
        self.assertEqual(chord.symbol, 'Abm')
        with self.assertRaises(err.InvalidChordNotesException):
            chord.notes = [13, 3, 6]
        with self.assertRaises(err.InvalidChordNotesException):
            chord.notes = 'B'
        with self.assertRaises(err.InvalidChordNotesException):
            chord.notes = []
            

    def test_set_symbol(self):
        """
        Test if the setter for the symbol is correct.
        """
        chord = Chord('Abmin')
        self.assertEqual(chord.notes, [Note(8), Note(11), Note(3)])
        chord.symbol = 'B'
        self.assertEqual(chord.symbol, 'B')
        self.assertEqual(chord.root, Note('B'))
        self.assertEqual(chord.triad, 'B')

    def test_cannot_wrongly_set_symbol(self):
        """
        Checks that wrongly formatted symbol can't be set by the constructor or the setter.
        """
        with self.assertRaises(err.InvalidChordSymbolException):
            chord = Chord('K')
        chord = Chord('Abmin')
        with self.assertRaises(err.InvalidChordSymbolException):
            chord.symbol = 1
        with self.assertRaises(err.InvalidChordSymbolException):
            chord.symbol = 'K'
    
    def test_cannot_modify_root(self):
        chord = Chord('A')
        with self.assertRaises(err.CantModifyAttribute):
            chord.root = Note('B')
    
    def test_add_note(self):
        """
        Checks the add_note() function.
        """
        chord = Chord([8, 0, 3])
        chord.add_note(6)
        self.assertTrue(chord == Chord([8, 0, 3, 6]))
        self.assertEqual(chord.symbol, cst.AMBIGUOUS)
        self.assertEqual(chord.root, cst.AMBIGUOUS)
        self.assertEqual(chord.triad, cst.AMBIGUOUS)
        self.assertTrue(chord[3] == Note(6))
        chord.add_note(8)
        self.assertTrue(chord == Chord([8, 0, 3, 6]))
        with self.assertRaises(err.InvalidNoteException):
            chord.add_note('K')

        redundant_chord = Chord([1, 5, 8], redundancy = True)
        redundant_chord.add_note(5)
        self.assertEqual(redundant_chord.get_numbers(), [1, 5, 8, 5])
    
    def test_get_nb_notes(self):
        """
        Checks the get_nb_notes() function.
        """
        chord = Chord([8, 0, 3])
        self.assertEqual(chord.get_nb_notes(), 3)
        chord = Chord([8, 0, 3, 6, 9])
        self.assertEqual(chord.get_nb_notes(), 5)
        chord = Chord([8, 0, 3, 8, 3, 0])
        self.assertEqual(chord.get_nb_notes(), 3)
        chord = Chord([8, 0, 3, 8, 3, 0], redundancy = True)
        self.assertEqual(chord.get_nb_notes(), 6)

    def test_vectorize(self):
        """
        Tests the vectorize() function.
        """
        self.assertTrue((Chord('Ab').vectorize() == np.array([1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.])).all())
        
        
##################################### Starting from the notes ################################
        
    def test_notes_to_symbol(self):
        """
        Tests if the symbol is correctly retrieved from the notes.
        """
        chord = Chord([8, 11, 3])
        symbol = 'Abm'
        self.assertEqual(chord.symbol, symbol)
        
        chord = Chord([8, 0, 3])
        symbol = 'Ab'
        self.assertEqual(chord.symbol, symbol)
        
        chord = Chord([8, 1, 3])
        self.assertEqual(chord.symbol, cst.AMBIGUOUS)
        self.assertEqual(chord.triad, cst.AMBIGUOUS)
        self.assertEqual(chord.root, cst.AMBIGUOUS)

        chord = Chord([8, 10, 3])
        self.assertEqual(chord.symbol, cst.AMBIGUOUS)
        self.assertEqual(chord.triad, cst.AMBIGUOUS)
        self.assertEqual(chord.root, cst.AMBIGUOUS)
        
    def test_root_from_notes(self):
        """
        Tests if the root is correctly retrieved from the notes.
        """
        chord = Chord([8, 11, 3])
        self.assertTrue(chord.root == Note(8))
        self.assertTrue(chord.root_from_notes() == Note(8))
        self.assertEqual(chord.root.number, 8)

        chord = Chord([3, 11, 8])
        self.assertTrue(chord.root == Note(8))

        chord = Chord([11, 3, 8])
        self.assertTrue(chord.root == Note(8))

        chord = Chord([3, 11, 8])
        self.assertTrue(chord.notes == [Note(8), Note(11), Note(3)])

    def test_triad_from_notes(self):
        """
        Tests if the triad is correctly retrieved from the notes.
        """
        chord_maj = Chord([8, 0, 3])
        chord_min = Chord([8, 11, 3])
        chord_sus4 = Chord([8, 1, 3])
        chord_sus2 = Chord([8, 10, 3])
        chord_aug = Chord([8, 0, 4])
        chord_dim = Chord([8, 0, 2])
        
        self.assertEqual(chord_maj.triad, "Ab")
        self.assertEqual(chord_min.triad, "Abm")
        self.assertEqual(chord_sus4.triad, cst.AMBIGUOUS)
        self.assertEqual(chord_sus2.triad, cst.AMBIGUOUS)
        self.assertEqual(chord_aug.triad, cst.AMBIGUOUS)
        self.assertEqual(chord_dim.triad, cst.AMBIGUOUS)
    

##################################### Starting from the symbol ################################

    def test_symbol_to_note(self):
        """
        Tests if the notes are correctly retrieved from the symbol.
        """
        chord_min = Chord('Abmin')
        notes = [8, 11, 3]
        self.assertEqual(chord_min.get_numbers(), notes)
        self.assertTrue(chord_min == Chord([8, 11, 3]))
        
        chord_maj = Chord('Abmaj')
        notes = [8, 0, 3]
        self.assertEqual(chord_maj.get_numbers(), notes)
        
        chord_maj = Chord('Ab5')
        notes = [8, 3]
        self.assertEqual(chord_maj.get_numbers(), notes)
        
        chord_sus4 = Chord('Absus4')
        notes = [8, 1, 3]
        self.assertEqual(chord_sus4.get_numbers(), notes)
        
        chord_sus2 = Chord('Absus2')
        notes = [8, 10, 3]
        self.assertEqual(chord_sus2.get_numbers(), notes)
        
        chord_dim = Chord('Abdim')
        notes = [8, 0, 2]
        self.assertEqual(chord_dim.get_numbers(), notes)
        
        chord_dim = Chord('Abd')
        notes = [8, 0, 2]
        self.assertEqual(chord_dim.get_numbers(), notes)
        
        chord_dim = Chord('Ab-')
        notes = [8, 0, 2]
        self.assertEqual(chord_dim.get_numbers(), notes)
        
        chord_aug = Chord('Abaug')
        notes = [8, 0, 4]
        self.assertEqual(chord_aug.get_numbers(), notes)
        
        chord_aug = Chord('Aba')
        notes = [8, 0, 4]
        self.assertEqual(chord_aug.get_numbers(), notes)
        
        chord_aug = Chord('Ab+')
        notes = [8, 0, 4]
        self.assertEqual(chord_aug.get_numbers(), notes)

        complex_chord = Chord('Ab7s2')
        notes = [8, 10, 3, 6]
        self.assertEqual(complex_chord.get_numbers(), notes)

        complex_chord = Chord('Abs27')
        notes = [8, 10, 3, 6]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('Cs2s4')
        notes = [0, 2, 5]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C7s2')
        notes = [0, 2, 7, 10]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C7s4')
        notes = [0, 5, 7, 10]
        self.assertEqual(complex_chord.get_numbers(), notes)

        complex_chord = Chord('C7s2s4')
        notes = [0, 2, 5, 10]
        self.assertEqual(complex_chord.get_numbers(), notes)

        complex_chord = Chord('Cm7')
        notes = [0, 3, 7, 10]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C7')
        notes = [0, 4, 7, 10]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C7d9')
        notes = [0, 4, 7, 10, 1]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C7+')
        notes = [0, 4, 8, 10]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C7-')
        notes = [0, 4, 6, 10]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C6')
        notes = [0, 4, 7, 9]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C9')
        notes = [0, 4, 7, 10, 2]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C9+')
        notes = [0, 4, 8, 10, 2]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C9-')
        notes = [0, 4, 6, 10, 2]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('C&9')
        notes = [0, 4, 7, 2]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('CM7')
        notes = [0, 4, 7, 11]
        self.assertEqual(complex_chord.get_numbers(), notes)
        
        complex_chord = Chord('CmM7')
        notes = [0, 3, 7, 11]
        self.assertEqual(complex_chord.get_numbers(), notes)


    def test_root_from_symbol(self):
        """
        Tests if the root is correctly retrieved from the symbol.
        """
        chord = Chord('Asus2')
        self.assertTrue(chord.root == Note('A'))
        chord = Chord('D')
        self.assertTrue(chord.root == Note('D'))

    def test_triad_from_symbol(self):
        """
        Tests if the triad is correctly retrieved from the symbol.
        """
        chord = Chord('A')
        self.assertEqual(chord.triad, "A")
        chord = Chord('Amaj')
        self.assertEqual(chord.triad, "A")
        chord = Chord('Am')
        self.assertEqual(chord.triad, "Am")
        chord = Chord('Amin')
        self.assertEqual(chord.triad, "Am")
        chord = Chord('Adim')
        self.assertEqual(chord.triad, "A")
        chord = Chord('Asus4')
        self.assertEqual(chord.triad, "A")
        chord = Chord('Asus2')
        self.assertEqual(chord.triad, "A")
        chord = Chord('A7')
        self.assertEqual(chord.triad, "A")
        chord = Chord('Am7')
        self.assertEqual(chord.triad, "Am")


if __name__ == '__main__':
    unittest.main()