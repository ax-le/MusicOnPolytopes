# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:06:19 2019

@author: amarmore
"""
class InvalidArgumentValueException(BaseException): pass
class OutdatedBehaviorException(BaseException): pass

class TriadException(BaseException): pass
class NotAMajMinTriadException(TriadException): pass

class NoteException(BaseException): pass
class InvalidNoteException(NoteException): pass
class InvalidNoteNumberException(InvalidNoteException): pass
class InvalidNoteSymbolException(InvalidNoteException): pass
class NotANoteObject(NoteException): pass

class ChordException(BaseException): pass
class InvalidChordException(ChordException): pass
class InvalidChordSymbolException(InvalidChordException): pass
class InvalidChordNotesException(InvalidChordException): pass
class CantModifyAttribute(ChordException): pass
class NotAChordObject(ChordException): pass
class NotATriadException(InvalidChordException): pass
class SymbolToDebugError(InvalidChordException): pass

class PolytopeException(BaseException): pass
class UnexpectedDim1Pattern(PolytopeException): pass
class UnexpectedDimensionForPattern(PolytopeException): pass
class PatternAndSequenceIncompatible(PolytopeException): pass

class IrregularPolytopeException(PolytopeException): pass
class WrongIrregularCode(IrregularPolytopeException): pass
class PatternToDebugError(IrregularPolytopeException): pass
class NoPolytopeForThisSize(IrregularPolytopeException): pass

class IndexException(PolytopeException): pass
class InvalidIndexException(IndexException): pass
class InvalidIndexSizeException(IndexException): pass
class ElementNotFound(IndexException): pass

class InvalidParameterException(BaseException): pass

class ToDebugException(BaseException): pass
