# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:20:03 2019

@author: amarmore
"""

#! /usr/bin/python
# -*- coding: utf-8 -*-
import polytopes.model.errors as err
from polytopes.model.constants import Constants as cst
from polytopes.model.note import Note
import polytopes.data_manipulation as dm
import numpy as np

# A Chord object
class Chord:
    """
    A class to compute chords as objects, with attributes such as their notes (note objects), their root, their symbol,
    and their triad type if it's a triad.
    
    Triads are better handled than other chords as they're non ambiguous chords (other chords can be defined with multiple symbols and roots). 
    
    A chord object doesn't allow repetition in its notes, except specified as argument in the constructor.
    """
# %% Defining the object attributes and tests
    def __init__(self, a_chord, redundancy = False):
        """
        Constructor for the Chord object.
        
        Parameters
        ----------
        a_chord: list of int or list of Note objects (see note.py)
            The notes for the chord to create.
                    OR
                string
            Symbol of the chord
                    !!!!!! Cannot initialize with lists of Note Symbols (ex: ['A', 'C', 'E']) (at least yet...)
        redundancy: boolean
            A boolean to allow (True) or disallow (False) the repetition of a same note in the Chord.
            The default comportement should disallow redundancy, as notes are considered as pitch classes, which repetition wouldn't add information.
            Still, it can be useful for the padding of Chords (notably in the computation of permutation and optimal trnasport between chords).
            Default: False
        Instanciates
        ------------
        A chord with:
            notes: list of Note objects
                List of all the notes forming the chord (Note objects, see note.py)
            root: Note object
                The root of the chord as a note object, if there is one.
                Starting from the symbol, the root will be the note defined by its dominant symbol.
                Starting from the notes, it will be defined in the cases of perfect major and minor chords, and set to "Ambiguous" otherwise (as the root is ambiguous in ohter types of chords.)
            triad: string or None
                The triad reduction of the chord.
                Every chord can be reduced to a triad when defined with a symbol, as it means taking the root as the definition of the triadic chord, with a possible minor sign.
                Starting from the notes, only perfect major and minor third will have a triad non labeled as "Ambiguous"
            symbol: string
                Symbol of the chord 
        """
        self._redundant = redundancy # To propagate the redundancy to the object, notably when modifying the notes after its creation.
        
        try: # If it's a tab of integers
            local_notes = []
            for a_note in a_chord:
                if dm.is_a_note_object(a_note):
                    note_obj = a_note
                else:
                    note_obj = Note(int(a_note))
                if not note_obj.is_in_list_of_notes(local_notes) or self._redundant:
                    local_notes.append(note_obj)
            self._notes = local_notes
            if self._notes == []:
                raise err.InvalidChordNotesException("Empty list of notes: a Chord must admit notes.") from None
            self._triad = self.triad_from_notes()
            if self.triad != cst.AMBIGUOUS and not self._redundant: # A redundant chord shouldn't be reordered as it corresponds to permutated chord (#TODO: rename the parameter ?)
                self._notes = self.triadic_order_notes()
            self._root = self.root_from_notes()
            self._symbol = self.notes_to_symbol()

        except ValueError: # If it's a symbol
            try:
                self._symbol = format_symbol(a_chord)
                self._root = self.root_from_symbol()
                self._triad = self.triad_from_symbol()
                self._notes = self.symbol_to_notes()
                if self._notes == []:
                    raise err.InvalidChordNotesException("Empty list of notes: a Chord must admit notes.") from None
            except err.NoteException as exc:
                raise err.InvalidChordSymbolException("Initialization of a Chord with an invalid symbol: {}.".format(a_chord)) from exc
            
        except err.NoteException as exc:
            raise err.InvalidChordNotesException("Initialization of a Chord with invalid notes: {}.".format(a_chord)) from exc

    def __repr__(self):
        """
        Defines a way to display the object when the repr() function is called, and its content.
        """
        return "%s(Symbol: %r, Notes: %r)" % (self.__class__, self.symbol, self.notes)

    def _get_chord_notes(self):
        """
        Getter for the notes of the chord.
        Returns
        -------
            List of Note objects:
                The notes of the chord
        """
        return self._notes
    
    def _set_chord_notes(self, notes):
        """
        Setter for the notes of the chord.
        Admits rules to correctly modify notes, and modify the other attributes of a chord accordingly.
        Parameters
        ----------
        notes: list of Note objects, integers or symbols.
            The new notes of the chord.
        Returns
        -------
        Updates the notes if they're correct and then updates the attributes of a chord (root, triad) and the symbol.
        """
        try:
            local_notes = []
            for a_note in notes:
                if dm.is_a_note_object(a_note):
                    note_obj = a_note
                else:
                    note_obj = Note(int(a_note))
                if not note_obj.is_in_list_of_notes(local_notes) or self._redundant:
                    local_notes.append(note_obj)
        except (ValueError, err.NoteException) as exc:
            raise err.InvalidChordNotesException("These notes are not valid notes") from exc
        self._notes = local_notes
        if self._notes == []:
            raise err.InvalidChordNotesException("Empty list of notes: a Chord must admit notes.") from None
        self._triad = self.triad_from_notes()
        if self.triad != cst.AMBIGUOUS and not self._redundant: # A redundant chord shouldn't be reordered as it corresponds to permutated chord (#TODO: rename the parameter ?)
            self._notes = self.triadic_order_notes()
        self._root = self.root_from_notes()
        self._symbol = self.notes_to_symbol()
    
    notes = property(_get_chord_notes, _set_chord_notes)
    
    def _get_chord_symbol(self):
        """
        Getter for the symbol of the chord.
        Returns:
            string: the symbol of the chord
        """
        return self._symbol
    
    def _set_chord_symbol(self, symbol):
        """
        Setter for the symbol of the chord.
        Admits rules to correctly modify the symbol, and modify the other attributes of a chord accordingly.
        Parameters
        ----------
        symbol: string
            The new symbol of the chord.
        Returns
        -------
        Updates the symbol and then updates the attributes of a chord (root, triad) and the notes.
        """
        try:
            self._symbol = format_symbol(symbol)
            self._root = self.root_from_symbol()
            self._triad = self.triad_from_symbol()
            self._notes = self.symbol_to_notes()
            if self._notes == []:
                raise err.InvalidChordNotesException("Empty list of notes: a Chord must admit notes.") from None
        except err.NoteException as exc:
            raise err.InvalidChordSymbolException("Initialization of a Chord with an invalid symbol.") from exc

    symbol = property(_get_chord_symbol, _set_chord_symbol)
    
    def _get_chord_root(self):
        """
        Getter for the root of the chord.
        Returns
        -------
        Note object: the root of the chord
        """
        return self._root
    
    def _get_chord_triad(self):
        """
        Getter for the triad of the chord.
        Returns
        -------
        string: the triad of the chord
        """
        return self._triad

    def _not_on_my_watch(self, *args, **kwargs):
        """
        Setter for the root and the triad.
        Raises an error, as these attributes must be found from either the notes or the symbol, and not set independantly (they don't contain enough information on the chord).
        """
        raise err.CantModifyAttribute("Not on my watch: you can't modify this attribute alone, you can only modify the notes or the symbol.") from None
    
    root = property(_get_chord_root, _not_on_my_watch)
    triad = property(_get_chord_triad, _not_on_my_watch)

    def __getitem__(self, i):
        """
        Overrides the chord[i] method.
        Parameters
        ----------
        i: integer:
            The index of the note to return.
        Returns
        -------
        Note object: the i-th note.
        """
        return self.notes[i]

    def __contains__(self, a_note):
        """
        Overrides the "note in chord" method.
        Parameters
        ----------
        a_note: Note object:
            the note to test whether it's contained by the chord.
        Returns
        -------
        boolean: True if the note can be found in the chord by its number (and not memory space as should be the default comportment), False if any note of the chord has this number.
        """
        return a_note.is_in_chord(self)

    def __eq__(self, another_chord):
        """
        Overrides the "chord == another_chord" test.
        Parameters
        ----------
        another_chord: Chord object:
            The other chord with which test the equality
        Returns
        -------
        boolean: True if both chords have exactly the same notes, False otherwise. Notes are considered equal by their numbers.
        """
        if self.get_nb_notes() != another_chord.get_nb_notes():
            return False
        for i, notes in enumerate(self.notes):
            if not notes == another_chord[i]:
                return False
        for i, notes in enumerate(another_chord.notes):
            if not notes == self[i]:
                return False
        return True

    """
    def __len__(self):
        return len(self.notes)
    
    def index(self, x):
        return self.notes.index(x)
    """

    def add_note(self, new_note):
        """
        Method to add a note to the notes attribute.
        Parameters
        ----------
        new_note: int, string or Note object
            The note to add to the notes attribute, as a Note object, or a number or a symbol of a note.
        Returns
        -------
        Updates the notes attribute and then updates the seventh, bass and symbol as they could be impacted by this addition.
        NB: this is true as chords are considered to be at least a triad.
        If this must change, update triad too.
        """
        try:
            if not dm.is_a_note_object(new_note):
                new_note = Note(new_note)
            notes = self.notes
            notes.append(new_note)
            self.notes = notes # Updates the entre object in the setter for the notes
        except err.NoteException:
            raise err.InvalidNoteException("Can't add a note to this chord as the note is not correct.")
            
# %% Defines the Chord object, starting from its notes
            
    def triad_from_notes(self):
        """
        Computes the triadic reduction of the chord from the notes.
        If the notes doesn't define a triad, it will be set as "Ambiguous", as no absolute triadic reduction can be deduced.
        Returns
        -------
        string: the triad of the chord. 
        """
        if len(self.notes) != 3:
            return cst.AMBIGUOUS
        for i in range(3):
            root = self.notes[i]
            first_gap = (self.notes[(i+1)%3].number - root.number)%12
            second_gap = (self.notes[(i+2)%3].number - root.number)%12
            if first_gap == 7:
                if second_gap == 3:
                    return root.symbol + "m"
                elif second_gap == 4:
                    return root.symbol
            elif second_gap == 7:
                if first_gap == 3:
                    return root.symbol + "m"
                elif first_gap == 4:
                    return root.symbol
        return cst.AMBIGUOUS

    def triadic_order_notes(self):
        """
        Order the notes in the order root - third- fifth if this is a triad.
        Returns
        -------
        list of Note objects: the notes ordered in the traidic order.
        """
        if len(self.notes) != 3:
            raise err.NotATriadException("Invalid number of notes: this is not a triad.")
        for i in range(3):
            root = self.notes[i]
            first_gap = (self.notes[(i+1)%3].number - root.number)%12
            second_gap = (self.notes[(i+2)%3].number - root.number)%12
            if first_gap == 7:
                if second_gap == 3:
                    return [root, self.notes[(i+2)%3], self.notes[(i+1)%3]]
                elif second_gap == 4:
                    return [root, self.notes[(i+2)%3], self.notes[(i+1)%3]]
            elif second_gap == 7:
                if first_gap == 3:
                    return [root, self.notes[(i+1)%3], self.notes[(i+2)%3]]
                elif first_gap == 4:
                    return [root, self.notes[(i+1)%3], self.notes[(i+2)%3]]
        raise err.NotATriadException("Invalid space between notes: this is not a triad.")
            
    def root_from_notes(self):
        """
        Computes the root from the notes.
        Returns
        -------
        Note: the root of the chord.
            or
        string: "Ambiguous" if its not a perfect major or minor chord.
        """
        if self.triad != cst.AMBIGUOUS:
            if not self._redundant:
                return self.notes[0]
            else:
                return self.triadic_order_notes()[0]
        else:
            return cst.AMBIGUOUS # or None
        
    def notes_to_symbol(self):
        """
        Computes the symbol of the chord from its notes.
        The only unambiguous chords from the notes are perfect major and mnor chords, so they're the only ones retrievable.
        Any other chord would be labeled as "Ambiguous".
        From the notes, this case is similar to the triad definition, starting from the notes.
        Returns
        -------
        chord_symbol: string
            the symbol of this chord if perfect major or minor
                or
            "Ambiguous" otherwise
        """
        return self.triad
    
            
# %% Defines the Chord object, starting from its symbol
    
    def root_from_symbol(self):
        """
        Computes the root from the symbol (which is the note symbol of the chord).
        Returns
        -------
        Note object: the root of the chord.
        """
        # if "*" in self.symbol:
        #     return None
        root_name = self.symbol[0]
        if len(self.symbol) > 1 and (self.symbol[1] == 'b' or self.symbol[1]== '#'):
            root_name += self.symbol[1] # Add sharp/flat symbol to the root note
        return Note(root_name)
        
    def triad_from_symbol(self):
        """
        Computes the triad reduction of the chord from the symbol.
        If it's a complex chord, the retrieved element will be the first three notes, defining a perfect major or minor chord.
        Returns
        -------
        string: the triad symbol of the chord.
        """
        idx_post_root = 1
        if len(self.symbol) > 1 and (self.symbol[1] == 'b' or self.symbol[1]== '#'):
            idx_post_root += 1
        
        if len(self.symbol) > idx_post_root and self.symbol[idx_post_root] == "m": # minor triad
            idx_post_root += 1
            
        return self.symbol[:idx_post_root]


    # %% Implementation of the musicological rules for finding notes from the symbol.
    def symbol_to_notes(self):
        """
        Computes all the notes from the symbol.
        This defines a huge function as numerous musicologial rules needs to be implemented.
        Returns
        -------
        list of Note objects: the notes of the chord.        
        """
        notes = []
        if "*" not in self.symbol:
            notes = [self.root]
        else:
            self.symbol = self.symbol.replace("*", "")
        
        # %% If simple triad (Perfect Major or Minor chord)
        if self.triad == self.symbol:
            if "m" in self.symbol:
                notes.append(Note((self.root.number+3)%12))
            else:
                notes.append(Note((self.root.number+4)%12))
            notes.append(Note((self.root.number+7)%12))
            return notes
        
        # %% 3 first notes
        idx_current_parsing = 1
        if self.symbol[1] in ["#", "b"]:
            idx_current_parsing += 1
            
        if self.symbol[idx_current_parsing] == "m": # Perfect Minor chord, with something after
            notes.append(Note((self.root.number+3)%12))
            notes.append(Note((self.root.number+7)%12))
            idx_current_parsing += 1
            
        elif self.symbol[idx_current_parsing] in ["d", "-"]: # Diminshed fifth
            notes.append(Note((self.root.number+4)%12))
            notes.append(Note((self.root.number+6)%12))
            idx_current_parsing += 1
            
        elif self.symbol[idx_current_parsing] in ["a", "+"]: # Augmented fifth
            notes.append(Note((self.root.number+4)%12))
            notes.append(Note((self.root.number+8)%12))
            idx_current_parsing += 1
            
        elif self.symbol[idx_current_parsing] == "s": # Suspended chords (just after the root symbol)
            if self.symbol[idx_current_parsing + 1] == '2':
                notes.append(Note((self.root.number+2)%12))
                if self.symbol[idx_current_parsing + 2:idx_current_parsing + 4] == 's4':
                    notes.append(Note((self.root.number+5)%12))
                    idx_current_parsing += 2
                else:
                    notes.append(Note((self.root.number+7)%12))
                idx_current_parsing += 2
            elif self.symbol[idx_current_parsing + 1] == '4':
                notes.append(Note((self.root.number+5)%12))
                notes.append(Note((self.root.number+7)%12))
                idx_current_parsing += 2
            else:
                raise err.SymbolToDebugError("Illegal value of suspended: " + self.symbol)
                
        elif self.symbol[idx_current_parsing] == "5": # Chord without third
            notes.append(Note((self.root.number+7)%12))
            idx_current_parsing += 1
            
        elif "s" in self.symbol: # Suspended chords (but suspended symbol somewhere in the chord, typically after the seventh symbol)
            idx_s = self.symbol.find("s")
            if self.symbol.count("s") == 2: # s2s4
                notes.append(Note((self.root.number+2)%12))
                notes.append(Note((self.root.number+5)%12))
            elif self.symbol[idx_s + 1] == '2':
                notes.append(Note((self.root.number+2)%12))
                notes.append(Note((self.root.number+7)%12))
            elif self.symbol[idx_s + 1] == '4':
                notes.append(Note((self.root.number+5)%12))
                notes.append(Note((self.root.number+7)%12))
            else:
                raise err.SymbolToDebugError("Illegal value of suspended: " + self.symbol)
                
        elif self.symbol[idx_current_parsing] == "M": # Perfect Major chord, with something after
            notes.append(Note((self.root.number+4)%12))
            notes.append(Note((self.root.number+7)%12))
            idx_current_parsing += 1

        else: # Perfect Major chord, with something after
            notes.append(Note((self.root.number+4)%12))
            notes.append(Note((self.root.number+7)%12))
            
        # %% Chords with more than 3 notes
        while len(self.symbol) != idx_current_parsing:
            # %% Symbol[idx_current_parsing] is int
            try:
                int(self.symbol[idx_current_parsing])
                
                if self.symbol[idx_current_parsing] == '6':
                    idx_current_parsing += 1
                    notes.append(Note((self.root.number+9)%12))
                    if len(self.symbol) != idx_current_parsing + 2 * self.symbol.count("s"):
                        # Is it possible to have something after the 6 symbol ?
                        raise err.SymbolToDebugError("Something after the sixth (not taken in account): " + self.symbol)
                    else:
                        idx_current_parsing += 2 * self.symbol.count("s")
                
                elif self.symbol[idx_current_parsing] == '7':
                    idx_current_parsing += 1
                    if len(self.symbol) == idx_current_parsing or self.symbol[idx_current_parsing] == "s":
                        notes.append(Note((self.root.number+10)%12))
                        if len(self.symbol) not in [idx_current_parsing, idx_current_parsing + 2, idx_current_parsing + 4]:
                            # Is it possible to have something after the suspended symbol ?
                            raise err.SymbolToDebugError("Unknown symbol: " + self.symbol)
                        return notes # To avoid a conditionnal incrementation of idx_parsing dependant of the suspended value
                    
                    elif self.symbol[idx_current_parsing] == "+": # Augmented triad, replace fifth
                        notes.remove(Note((self.root.number+7)%12))
                        notes.append(Note((self.root.number+8)%12))
                        notes.append(Note((self.root.number+10)%12))
                        idx_current_parsing += 1
                        
                    elif self.symbol[idx_current_parsing] == '-': # Diminished triad, replace fifth
                        notes.remove(Note((self.root.number+7)%12))
                        notes.append(Note((self.root.number+6)%12))
                        notes.append(Note((self.root.number+10)%12))
                        idx_current_parsing += 1
                    
                    elif self.symbol[idx_current_parsing] == 'd':
                        idx_current_parsing += 1
                        if self.symbol[idx_current_parsing] == '9':
                            notes.append(Note((self.root.number+10)%12))
                            notes.append(Note((self.root.number+1)%12))
                            idx_current_parsing += 1
                        else:
                            # What is used with 7d other than 7d9 ?
                            raise err.SymbolToDebugError("Unknown symbol: " + self.symbol)
                    else:
                        # What else happen after a 7 ?
                        raise err.SymbolToDebugError("Unknown symbol: " + self.symbol)
                            
                elif self.symbol[idx_current_parsing] == '9':
                    idx_current_parsing += 1

                    if len(self.symbol) == idx_current_parsing:
                        notes.append(Note((self.root.number + 10)%12)) # Seventh associated with the ninth
                        notes.append(Note((self.root.number + 2)%12))
                    
                    elif self.symbol[idx_current_parsing] == "+" or self.symbol[idx_current_parsing] == "a": # Augmented ninth
                        notes.remove(Note((self.root.number+7)%12))
                        notes.append(Note((self.root.number+8)%12))
                        notes.append(Note((self.root.number+10)%12)) # Seventh associated with the ninth
                        notes.append(Note((self.root.number+2)%12))
                        idx_current_parsing += 1
                        
                    elif self.symbol[idx_current_parsing] == '-' or self.symbol[idx_current_parsing] == 'd': # Diminished ninth
                        notes.remove(Note((self.root.number+7)%12))
                        notes.append(Note((self.root.number+6)%12))
                        notes.append(Note((self.root.number+10)%12)) # Seventh associated with the ninth
                        notes.append(Note((self.root.number+2)%12))
                        idx_current_parsing += 1
                        
                    else:
                        # Something after the 9 ?
                        raise err.SymbolToDebugError("Unknown symbol " + self.symbol)

                else:
                    # What other int can be present other than 6, 7 and 9 ? (11th and 13th not evaluated)
                    raise err.SymbolToDebugError("Unknown symbol: " + self.symbol)
                
            # %% Symbol[idx_current_parsing] is not int
            except ValueError: 
                if self.symbol[idx_current_parsing] == '&':
                    idx_current_parsing += 1
                    if self.symbol[idx_current_parsing] == '9':
                        notes.append(Note((self.root.number + 2)%12))
                        idx_current_parsing += 1
                        if len(self.symbol) != idx_current_parsing:
                            # Nothing should happen after &9
                            raise err.SymbolToDebugError("Unknown symbol: " + self.symbol)
                        #return notes
                    elif self.symbol[idx_current_parsing:idx_current_parsing+2] == 'd9':
                        notes.append(Note((self.root.number + 2)%12))
                        idx_current_parsing += 2
                        if len(self.symbol) != idx_current_parsing:
                            # Nothing should happen after &d9
                            raise err.SymbolToDebugError("Unknown symbol: " + self.symbol)
                        #return notes
                    else:
                        # What is used with & ither than 9 ?
                        raise err.SymbolToDebugError("Unknown symbol: " + self.symbol)
    
                elif self.symbol[idx_current_parsing] == 'M':
                    idx_current_parsing += 1
                    if self.symbol[idx_current_parsing] == '7':
                        notes.append(Note((self.root.number+11)%12))
                        idx_current_parsing += 1
                    else:
                        # What comes after a M ?
                        raise err.SymbolToDebugError("Unknown symbol " + self.symbol)
                        
                else:
                    # What other symbol than a M or a & ?
                    raise err.SymbolToDebugError("Unknown symbol: " + self.symbol)
                    
        return notes

# %% Additionnary functions, which won't alter the object
        
    def is_in_list_of_chords(self, list_of_chords):
        """
        Test whether this chord is contained in a list of chords.
        Parameters
        ----------
        list_of_chords: list of Chords
            the list of chords to test whether it contains this (self) chord.
        Returns
        -------
        boolean: True if this chord is in the list, False otherwise.
        """
        for chord in list_of_chords:
            if self == chord:
                return True
        return False

    def get_nb_notes(self):
        """
        Encapsulation of the length of the chord in term of number of notes.
        Returns
        -------
        int: the number of notes in the chord.
        """
        return len(self.notes)
    
    def get_numbers(self):
        """
        A function to return the chord as a list of integers.
        Returns
        -------
        list of int: the number of all notes in the chord.
        """
        to_return = []
        for a_note in self.notes:
            to_return.append(a_note.number)
        return to_return

    def vectorize(self):
        """
        A function to get the chord as a 12-dimensional binary vector.
        The 12 dimensions corresponds to the 12 note numbers.
        The vector will contain a one at the indexes corresponding to a present note, and zeros at the other indexes.
        
        Returns
        -------
        np.array: the chord vectorized.
        """
        vec = np.zeros(12)
        #vec = np.zeros(12).reshape(1,12)
        for note in self.notes:
            vec[note.number] = 1
            #vec[:, note.number] = 1
        return vec
        
    
    def to_string(self, numbers = False):
        """
        Computes the chord (notes or symbol) as a string.
        Parameters
        ----------
        numbers: boolean
            True to return the notes of the chord as their numbers, False to return the symbol.
        Returns
        -------
        string: the numbers of the notes in the chord or the symbol of the chord.
        """
        if numbers:
            return str(self.get_numbers())
        else:
            return self.symbol

def format_symbol(symbol):
    """
    Rules for a correct symbol of the chord.
    Used when the chord is instanciated by its symbol, to avoid variances between symbols.
    Basically, the desired format is "root + triad + suppl", and aims at replacing a symbol 'Am' by 'Amin' for example.
    
    This function isn't defined in the chord object as it does not need to have access to the chord attributes.
    Can be put in the shared_functions module, #TODO: to think.
    
    Parameters
    ----------        
    symbol: string
        The desired symbol
    
    Returns
    -------
    symbol: the symbol transcribed in the desired format.
    """
    try:
        int(symbol)
        raise err.InvalidChordSymbolException("The symbol isn't valid.") from None
    except ValueError:
        symbol = symbol.replace(":","")
        symbol = symbol.replace("maj", "")
        symbol = symbol.replace("min", "m")
        symbol = symbol.replace("aug", "a")
        symbol = symbol.replace("dim", "d")
        symbol = symbol.replace("sus", "s")
        return symbol