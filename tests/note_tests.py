# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:03:20 2019

@author: amarmore
"""

import unittest
import polytopes.data_manipulation as dm
from polytopes.model.note import Note
import polytopes.model.errors as err

class NoteTests(unittest.TestCase):
    notes_to_integer = (('C',0),('D',2),('E',4),('F',5),('G',7),('A',9),('B',11))
    flat_notes_to_integer = (('Db',1),('Eb',3),('Gb',6),('Ab',8),('Bb',10))
    sharp_notes_to_integer = (('C#',1),('D#',3),('F#',6),('G#',8),('A#',10))

    def test_is_a_note_object(self):
        """
        Checks if the "shared.is_a_note_object()" function correctly recognizes Note objects.
        """
        self.assertTrue(dm.is_a_note_object(Note(1)))
        self.assertFalse(dm.is_a_note_object(1))

    def test_int_from_int_object(self):
        """
        Creates a Note object with its number, and checks if its attribute "number" is correct.
        """
        for symbol, number in self.notes_to_integer:
            a_note = Note(number)
            self.assertEqual(number, a_note.number)
        for symbol, number in self.flat_notes_to_integer:
            a_note = Note(number)
            self.assertEqual(number, a_note.number)
        for symbol, number in self.sharp_notes_to_integer:
            a_note = Note(number)
            self.assertEqual(number, a_note.number)
            
    def test_symbol_from_symbol_object(self):
        """
        Creates a Note object with its symbol, and checks if its attribute "symbol" is correct.
        The symbol differs between sharp and flat, and both cases symbols should be different.
        """
        for symbol, number in self.notes_to_integer:
            a_note = Note(symbol)
            self.assertEqual(symbol, a_note.symbol)
            
        for flat_symbol, number in self.flat_notes_to_integer:
            # Correctly flat
            a_note_flat = Note(flat_symbol, flat = True)
            self.assertEqual(flat_symbol, a_note_flat.symbol)
            # Wrongly sharp
            a_note_sharp = Note(flat_symbol, flat = False)
            self.assertNotEqual(flat_symbol, a_note_sharp.symbol)
            
        for sharp_symbol, number in self.sharp_notes_to_integer:
            # Correctly sharp
            a_note_sharp = Note(sharp_symbol, flat = False)
            self.assertEqual(sharp_symbol, a_note_sharp.symbol)
            # Wrongly flat
            a_note_flat = Note(sharp_symbol, flat = True)
            self.assertNotEqual(sharp_symbol, a_note_flat.symbol)

    def test_number_from_symbol_function(self):
        """
        Creates a Note object with its symbol, and checks if the function "number_from_symbol()" returns the correct number.
        """
        for symbol, number in self.notes_to_integer:
            a_note = Note(symbol)
            self.assertEqual(number, a_note.number_from_symbol())
        for symbol, number in self.flat_notes_to_integer:
            a_note = Note(symbol)
            self.assertEqual(number, a_note.number_from_symbol())
        for symbol, number in self.sharp_notes_to_integer:
            a_note = Note(symbol)
            self.assertEqual(number, a_note.number_from_symbol())
    
    def test_number_from_symbol_object(self):
        """
        Creates a Note object with its symbol, and checks if its attribute "number" is the correct number.
        The flat and sharp cases (when creating the note) should return the same number.
        """
        for symbol, number in self.notes_to_integer:
            a_note = Note(symbol)
            self.assertEqual(number, a_note.number)
        for symbol, number in self.flat_notes_to_integer:
            a_note = Note(symbol)
            self.assertEqual(number, a_note.number)
        for symbol, number in self.sharp_notes_to_integer:
            a_note = Note(symbol)
            self.assertEqual(number, a_note.number)
        
        for i in range(len(self.sharp_notes_to_integer)): # is it the same number between sharp and flat ?
            a_note_flat = Note(self.flat_notes_to_integer[i][0])
            a_note_sharp = Note(self.sharp_notes_to_integer[i][0])
            self.assertEqual(a_note_sharp.number, a_note_flat.number)

    def test_symbol_from_number_function(self):
        """
        Creates a Note object with its number, and checks if the function "symbol_from_number()" returns the correct symbol.
        The flat and sharp cases should be different and return different symbols.
        """
        for symbol, number in self.notes_to_integer:
            a_note = Note(number)
            self.assertEqual(symbol, a_note.symbol_from_number())
            
        for flat_symbol, number in self.flat_notes_to_integer:
            a_note = Note(number)
            self.assertEqual(flat_symbol, a_note.symbol_from_number(flat = True)) # is flat flat
            self.assertNotEqual(flat_symbol, a_note.symbol_from_number(flat = False)) # is sharp flat
            
        for sharp_symbol, number in self.sharp_notes_to_integer:
            a_note = Note(number)
            self.assertEqual(sharp_symbol, a_note.symbol_from_number(flat = False)) # is sharp sharp
            self.assertNotEqual(sharp_symbol, a_note.symbol_from_number(flat = True)) # is flat sharp
    
    def test_symbol_from_number_object(self):
        """
        Creates a Note object with its number, and checks if its attribute "symbol" is the correct symbol.
        The flat and sharp cases should be different and return different symbols.
        """
        for symbol, number in self.notes_to_integer:
            a_note = Note(number)
            self.assertEqual(symbol, a_note.symbol)
            
        for flat_symbol, number in self.flat_notes_to_integer:
            a_note_flat = Note(number, flat = True)
            self.assertEqual(flat_symbol, a_note_flat.symbol)
            a_note_sharp = Note(number, flat = False)
            self.assertNotEqual(flat_symbol, a_note_sharp.symbol)
            
        for sharp_symbol, number in self.sharp_notes_to_integer:
            a_note_sharp = Note(number, flat = False)
            self.assertEqual(sharp_symbol, a_note_sharp.symbol)
            a_note_flat = Note(number, flat = True)
            self.assertNotEqual(sharp_symbol, a_note_flat.symbol)

    def test_cannot_init_with_invalid_symbol(self):
        """
        Checks that we can't initialize a note with an incorrect symbol.
        """
        #self.assertRaises(NotImplementedError, lambda: Note('K'))
        self.assertRaises(err.InvalidNoteSymbolException, Note, 'K')
    
    def test_cannot_init_with_invalid_number(self):
        """
        Checks that we can't initialize a note with an incorrect number.
        """
        self.assertRaises(err.InvalidNoteNumberException, lambda: Note(13))

    def test_cannot_init_with_negatives_number(self):
        """
        Checks that we can't initialize a note with a negative number.
        """
        self.assertRaises(err.InvalidNoteNumberException, lambda: Note(-1))
        
    def test_cannot_modify_with_invalid_symbol(self):
        """
        Checks that we can't modify a note with an incorrect symbol.
        """
        a_note = Note(1)
        with self.assertRaises(err.InvalidNoteSymbolException):
            a_note.symbol = 'K'
        with self.assertRaises(err.InvalidNoteSymbolException):
            a_note.symbol = 1
            
    def test_cannot_modify_with_invalid_number(self):
        """
        Checks that we can't modify a note wirh an incorrect number.
        """
        a_note = Note(1)
        with self.assertRaises(err.InvalidNoteNumberException):
            a_note.number = 13
        with self.assertRaises(err.InvalidNoteNumberException):
            a_note.number = 'A'

    def test_cannot_modify_with_negative_number(self):
        """
        Checks that we can't modify a note wirh an incorrect number.
        """
        a_note = Note(1)
        with self.assertRaises(err.InvalidNoteNumberException):
            a_note.number = -1
            
    def test_symbol_modified_when_modifying_symbol(self):
        """
        Checks if the symbol is modified when symbol attribute is modified.
        """
        symbol_init, number_init = self.notes_to_integer[0]
        note = Note(symbol_init)
        new_symbol, new_number = self.notes_to_integer[1]
        note.symbol = new_symbol
        self.assertEqual(note.symbol, new_symbol)
        self.assertNotEqual(note.symbol, symbol_init)
        
    def test_number_modified_when_modifying_number(self):
        """
        Checks if the number is modified when number attribute is modified.
        """
        symbol_init, number_init = self.notes_to_integer[0]
        note = Note(number_init)
        new_symbol, new_number = self.notes_to_integer[1]
        note.number = new_number
        self.assertEqual(note.number, new_number)
        self.assertNotEqual(note.number, number_init)
            
    def test_symbol_modified_when_modifying_number(self):
        """
        Checks if the symbol is modified when number attribute is modified.
        """
        symbol_init, number_init = self.notes_to_integer[0]
        note = Note(symbol_init)
        new_symbol, new_number = self.notes_to_integer[1]
        note.number = new_number
        self.assertEqual(note.symbol, new_symbol)
        self.assertNotEqual(note.symbol, symbol_init)
        
    def test_number_modified_when_modifying_symbol(self):
        """
        Checks if the number is modified when symbol attribute is modified.
        """
        symbol_init, number_init = self.notes_to_integer[0]
        note = Note(number_init)
        new_symbol, new_number = self.notes_to_integer[1]
        note.symbol = new_symbol
        self.assertEqual(note.number, new_number)
        self.assertNotEqual(note.number, number_init)
    
    def test_sharp_to_flat(self):
        """
        Checks that we correctly convert a note from sharp to flat with to_flat() method.
        Checked with init by number and symbol.
        """
        for symbol, number in self.flat_notes_to_integer:
            a_number = Note(number, flat = False) # Created as a sharp
            a_number.to_flat()
            self.assertEqual(symbol, a_number.symbol)
            
            a_symbol = Note(symbol, flat = False) # Created as a sharp
            a_symbol.to_flat()
            self.assertEqual(symbol, a_symbol.symbol)

    def test_flat_to_sharp(self):
        """
        Checks that we correctly convert a note from flat to sharp with to_sharp() method.
        Checked with init by number and symbol.
        """
        for symbol, number in self.sharp_notes_to_integer:
            a_number = Note(number, flat = True) # Created as a flat
            a_number.to_sharp()
            self.assertEqual(symbol, a_number.symbol)
            
            a_symbol = Note(symbol, flat = True) # Created as a flat
            a_symbol.to_sharp()
            self.assertEqual(symbol, a_symbol.symbol)
    
    def test_function_eq_override(self):
        """
        Checks that the override of equals method on a note correctly compare notes.
        """
        basis_note = Note(self.notes_to_integer[0][0])
        
        same_note = Note(self.notes_to_integer[0][0]) # same note as before, but different object
        self.assertTrue(basis_note == same_note)
        self.assertTrue(same_note == basis_note)
        
        same_note_diff_init = Note(self.notes_to_integer[0][1]) # same note as before, but init with its number
        self.assertTrue(basis_note == same_note_diff_init)
        self.assertTrue(same_note_diff_init == basis_note)
        
        diff_note = Note(self.notes_to_integer[1][0]) # Different note from basis_note
        self.assertFalse(basis_note == diff_note)
        self.assertFalse(diff_note == basis_note)

if __name__ == '__main__':
    unittest.main()