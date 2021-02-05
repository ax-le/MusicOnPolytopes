# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:46:17 2019

@author: amarmore
"""
# A class used to manipulate Notes as symbolic and integer representation.

import polytopes.model.errors as err

import polytopes.data_manipulation as dm

class Note:
    """
    A class to compute and handle notes as objects, with two attributes:
        - their number, between 0 (for C) and 11 (for B)
        - their symbol (A,B,C,D,E,F,G with sharp and flat symbols).
    """
    
    def __init__(self, a_note, flat = True):
        """
        Constructor for the note object.
        
        Parameters
        ----------
        a_note: string or int
            The note to create.
        flat: boolean
            Whereas the symbol should be in the flat (True) or sharp (False) convention (if necessited).
        
        Instanciates
        ------------
        Note object, with:
            Symbol: string
                The symbol of the note.
            Number: int
                The number of the note.
        """
        try:
            # a_note should be its number
            note_number = int(a_note)
            if note_number >= 0:
                self._number = note_number
                self._symbol = self.symbol_from_number(flat = flat)
            else: # The number of the note can't be negative, otherwise it still instanciate an object due to side effects of lists, but with an incorrect number.
                raise err.InvalidNoteNumberException("The desired number is negative: can't correspond to a note.") from None

        except ValueError:
            # a_note should be the symbol
            if a_note in self.get_notes_sharp() or a_note in self.get_notes_flat():
                self._symbol = a_note
                self._number = self.number_from_symbol()
                if flat:
                    self.to_flat()
                else:
                    self.to_sharp()
            else:
                raise err.InvalidNoteSymbolException("This symbol is not a correct note symbol: {}.".format(a_note)) from None
        except IndexError: # If symbol_from_number raises an index error, it means that the number is out of the bounds.
            raise err.InvalidNoteNumberException("The desired number is too large to be a valid note.") from None
    
    def _get_symbol(self):
        """
        Getter for the note symbol.
        Returns
        -------
        The symbol of the note.
        """
        return self._symbol
    
    def _set_symbol(self, symbol):
        """
        Setter for note symbol.
        Admits rules on the accepted values.
        Once the symbol is instanciated, it modifies its number.
        
        Parameters
        ----------
        symbol: string
            The new symbol.
        Returns
        -------
        Updates the note with the new symbol and the number associated.
        """
        if symbol in self.get_notes_sharp() or symbol in self.get_notes_flat():
            self._symbol = symbol
            self._number = self.number_from_symbol()
        else:
            raise err.InvalidNoteSymbolException("This symbol is not a correct note symbol.") from None
    
    def _get_number(self):
        """
        Getter for number.
        Returns
        -------
        The number of the note.
        """
        return self._number
    
    def _set_number(self, number):
        """
        Setter for number.
        Admits rules on the accepted values.
        Once the number is instanciated, it modifies its symbol.
        
        Parameters
        ----------
        number: int (between 0 and 11)
            The new number.
        Returns
        -------
        Updates the note with the new number and the symbol associated.
        """
        try:
            note_number = int(number)
            if note_number >= 0:
                self._number = note_number
                self._symbol = self.symbol_from_number()
            else:
                raise err.InvalidNoteNumberException("The desired number is negative: can't correspond to a note.") from None
        except ValueError:
            raise err.InvalidNoteNumberException("The desired new number is not an integer, and is not valid.") from None
        except IndexError:
            raise err.InvalidNoteNumberException("The desired new number is too large to be a valid note.") from None

    
    # Allows to be more rigid on the condition of modification, and to modify number and symbol at the same time when modifying one.
    symbol = property(_get_symbol, _set_symbol)
    number = property(_get_number, _set_number)

    def __repr__(self):
        """
        Defines a way to display the object when the repr() function is called, and its content.
        """
        return "%s(Symbol: %r, Number: %r)" % (self.__class__, self.symbol, self.number)
    
    def __eq__(self, another_note):
        """
        Tests if the current note (self) is the same note (in term of value, not object) than another_note.
        It overrides the traditionnal "==" test, in order to compare the numbers and not the objects, in terms of memory index.
        
        Parameters
        ----------
        another_note: Note object
            The other note with which to compare.
        Returns
        -------
        True if both notes have the same number, False otherwise.
        """
        if not dm.is_a_note_object(another_note):
            #another_note = Note(another_note)
            raise err.NotANoteObject("The comparaison must be done between two note objects.") from None
        return self.number == another_note.number

    def to_flat(self):
        """
        Modify symbol to be in the flat convention.
        Returns
        -------
        Updates the symbol of the note in the flat convention.
        """
        self.symbol = Note.get_notes_flat()[self.number]
    
    def to_sharp(self):
        """
        Modify symbol to be in the sharp convention.
        Returns
        -------
        Updates the symbol of the note in the sharp convention.
        """
        self.symbol = Note.get_notes_sharp()[self.number]

    @staticmethod
    def get_notes_sharp():
        """
        All the symbols of notes in the sharp convention.
        Returns
        -------
        List of the symbols.
        """
        return ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                
    @staticmethod  
    def get_notes_flat():
        """
        All the symbols of notes in the flat convention.
        Returns
        -------
        List of the symbols.
        """
        return ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
        
    def is_in_list_of_notes(self, list_of_notes):
        """
        Tests if the current note (self) is in list_of_notes (which is a list of Note objects).
        This function is used to create a chord.
        
        Parameters
        ----------
        list_of_notes: List of Note objects
            The list in which is or isn't the note.
        Returns
        -------
        True if the note is found in the list (if a note has the same number), False otherwise.
        """
        for a_note in list_of_notes:
            if self == a_note:
                return True
        return False
    
    def is_in_chord(self, chord):
        """
        Tests if the current note (self) is in the chord (a Chord object, see chord.py).
        This function is used to test when the chord is already created.
        
        Parameters
        ----------
        chord: Chord object (see chord.py)
            The chord in which is or isn't the note.
        Returns
        -------
        True if the note is found in the chord (if a note has the same number), False otherwise.
        """
        if not dm.is_a_chord_object(chord):
            raise err.NotAChordObject("Incorrect parameter: is_in_chord takes a chord object as parameter.") from None
        for a_note in chord.notes:
            if self == a_note:
                return True
        return False

    def number_from_symbol(self):
        """
        Find the number of the note, from the symbol.
        Returns
        -------
        int: the number of the note.
        """
        notes_sharp = Note.get_notes_sharp()
        notes_flat = Note.get_notes_flat()
        for i in range(12):
            if notes_sharp[i] == self.symbol or notes_flat[i] == self.symbol:
                return i
        raise err.InvalidNoteException("Not a valid note to convert") from None
    
    def symbol_from_number(self, flat = True):
        """
        Find the symbol of the note, from the number.
        
        Parameters
        ----------
        flat: boolean
            Whereas the symbol should be in the flat (True) or sharp (False) convention (if necessited).
        Returns
        -------
        string: the symbol of the note.
        """
        if flat:
            notes = Note.get_notes_flat()
        else:
            notes = Note.get_notes_sharp()
        return notes[self.number]
    
    def to_string(self, number = True):
        """
        Return a note attribute (number or symbol) in a string.
        
        Parameters
        ----------
        number: boolean
            Whereas the output should be the number (True) or the symbol (False) of the note.
        Returns
        -------
        The number or the symbol, on a string.
        """
        if number:
            return str(self.number)
        else:
            return self.symbol
