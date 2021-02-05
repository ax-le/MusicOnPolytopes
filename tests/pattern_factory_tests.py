# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:05:47 2019

@author: amarmore
"""
import unittest

import polytopes.pattern_factory as pf
from polytopes.model.chord import Chord
import polytopes.model.errors as err
import polytopes.data_manipulation as dm
import math

class PolyopTests(unittest.TestCase):
    
    def setUp(self):
        self.chord_object_sequence_global = [Chord('A'),Chord('B'),Chord('C'),Chord('D'),Chord('E'),Chord('F'),Chord('G'),Chord('A#'),Chord('C#'),Chord('D#'),
                    Chord('F#'),Chord('G#'),Chord('A'),Chord('B'),
                    Chord('C'),Chord('D'),Chord('E'),
                    Chord('F'),Chord('G'),Chord('A#'),Chord('C#'),Chord('D#'),Chord('F#'),Chord('G#'),
                    Chord('A'),Chord('B'),Chord('C'),Chord('D'),
                    Chord('E'),Chord('F'),Chord('G'),Chord('A#'),Chord('C#'),Chord('D#'),
                    Chord('F#'),Chord('G#'),Chord('A'),Chord('B'),
                    Chord('C'),Chord('D'),Chord('E'),
                    Chord('F'),Chord('G'),Chord('A#'),Chord('C#'),Chord('D#'),Chord('F#'),Chord('G#'),
                    Chord('A'),Chord('B'),Chord('C'),Chord('D'),
                    Chord('E'),Chord('F'),Chord('G'),Chord('A#'),Chord('C#'),Chord('D#'),
                    Chord('F#'),Chord('G#'),Chord('A'),Chord('B'),
                    Chord('C'),Chord('D'),Chord('E'),
                    Chord('F'),Chord('G'),Chord('A#'),Chord('C#'),Chord('D#'),Chord('F#'),Chord('G#'),
                    Chord('A'),Chord('B'),Chord('C'),Chord('D'),
                    Chord('E'),Chord('F'),Chord('G'),Chord('A#'),Chord('C#'),Chord('D#'),
                    Chord('F#'),Chord('G#'),Chord('A'),Chord('B'),
                    Chord('C'),Chord('D'),Chord('E'),
                    Chord('F'),Chord('G'),Chord('A#'),Chord('C#'),Chord('D#'),Chord('F#'),Chord('G#')]
        
        self.chord_sequence_global = [self.chord_object_sequence_global[j].symbol for j in range(len(self.chord_object_sequence_global))]
            
    def test_get_pattern_dimension(self):
        """
        Checks function get_pattern_dimension().
        """
        a_pattern = pf.make_regular_polytope_pattern(3)
        self.assertEqual(pf.get_pattern_dimension(a_pattern), 3)
        self.assertNotEqual(pf.get_pattern_dimension(a_pattern), 2)

        a_pattern = pf.make_polytope_pattern(2, adding_code = [], deleting_code = [])
        self.assertEqual(pf.get_pattern_dimension(a_pattern), 2)
        self.assertNotEqual(pf.get_pattern_dimension(a_pattern), 3)
        
        a_pattern = pf.make_polytope_pattern(3, adding_code = [], deleting_code = [])
        self.assertEqual(pf.get_pattern_dimension(a_pattern[0]), 2)
        
        a_pattern = pf.make_polytope_pattern(3, adding_code = [1,0,0], deleting_code = [])
        self.assertNotEqual(pf.get_pattern_dimension(a_pattern), 2)

        a_pattern = pf.make_polytope_pattern(3, adding_code = [1,0,0], deleting_code = [0,1,1])
        self.assertEqual(pf.get_pattern_dimension(a_pattern), 3)
        
    def test_cannot_wrongly_set_polytope_pattern(self):
        with self.assertRaises(err.WrongIrregularCode):
            a_pattern = pf.make_polytope_pattern(3, adding_code = [1,0,0,1], deleting_code = [])
        
    def test_get_size(self):
        a_polytope_pattern = pf.make_polytope_pattern(3, adding_code = [], deleting_code = [])
        self.assertEqual(pf.get_pattern_size(a_polytope_pattern), 2**3)

        a_polytope_pattern = pf.make_polytope_pattern(3, adding_code = [], deleting_code = [])
        self.assertEqual(pf.get_pattern_size(a_polytope_pattern), 2**3)

        a_polytope_pattern = pf.make_polytope_pattern(3, adding_code = [0,0,1], deleting_code = [])
        self.assertEqual(pf.get_pattern_size(a_polytope_pattern), 10)
        
        a_polytope_pattern = pf.make_polytope_pattern(4, adding_code = [0,0,1,1], deleting_code = [1,0,0,1])
        self.assertEqual(pf.get_pattern_size(a_polytope_pattern), 14)
        
        for size in range(6, 65):
            dim = round(math.log(size,2))
            for add, dele in pf.get_codes(size):
                self.assertEqual(size, pf.get_pattern_size(pf.make_polytope_pattern(dim, adding_code = add, deleting_code = dele)))


    def test_generate_irreg_codes(self):
        self.assertEqual(pf.generate_irreg_codes(3,2),
                         [[],[0, 0, 0],[0, 0, 1],[0, 1, 0],[0, 1, 1],[1, 0, 0],[1, 0, 1],[1, 1, 0]])
        self.assertEqual(pf.generate_irreg_codes(4,2),
                         [[], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0]])
        
    def test_get_deformation_size(self):
        self.assertEqual(pf.get_deformation_size([1,0,0,1]),4)
        self.assertEqual(pf.get_deformation_size([1,0,1,1]),8)
        self.assertEqual(pf.get_deformation_size([0,0,0,0]),1)
        self.assertEqual(pf.get_deformation_size([0,0,0,1]),2)
        self.assertEqual(pf.get_deformation_size([0,0,1,1,1,1]),16)

    def test_get_final_polytope_size(self):
        self.assertEqual(pf.get_final_pattern_size(3,[0, 0, 0],[0, 0, 1]), 6)
        self.assertEqual(pf.get_final_pattern_size(4,[1, 0, 0, 0],[0,0, 0, 1]), 15)
        self.assertEqual(pf.get_final_pattern_size(4,[1, 0, 1, 0],[0, 0, 0, 1]), 17)
        self.assertEqual(pf.get_final_pattern_size(5,[1, 0,0, 0, 0],[1,0, 0, 0, 1]), 28)
        self.assertEqual(pf.get_final_pattern_size(2,[0,1],[]), 6)
        self.assertEqual(pf.get_final_pattern_size(2,[1,0],[0,0]), 4)
        
    def test_get_codes(self):
        self.assertEqual(pf.get_codes(9),
                         [([0, 0, 0], [])])
        self.assertEqual(pf.get_codes(13),
                         [([0, 0, 0, 1], [0, 1, 1, 0]), ([0, 0, 0, 1], [1, 0, 1, 0]), ([0, 0, 0, 1], [1, 1, 0, 0]), ([0, 0, 1, 0], [0, 1, 0, 1]), ([0, 0, 1, 0], [1, 0, 0, 1]),
                         ([0, 0, 1, 0], [1, 1, 0, 0]), ([0, 1, 0, 0], [0, 0, 1, 1]), ([0, 1, 0, 0], [1, 0, 0, 1]), ([0, 1, 0, 0], [1, 0, 1, 0]), ([1, 0, 0, 0], [0, 0, 1, 1]), ([1, 0, 0, 0], [0, 1, 0, 1]), ([1, 0, 0, 0], [0, 1, 1, 0])])
    
        self.assertEqual(pf.get_codes(11),
                         [])

    def test_get_unique_codes(self):
        self.assertEqual(pf.get_unique_codes(12),
                         [([], [0, 0, 1, 1]), ([], [0, 1, 0, 1]), ([], [0, 1, 1, 0]), ([], [1, 0, 0, 1]), ([], [1, 0, 1, 0]), ([], [1, 1, 0, 0])])
        
    def test_get_pattern_size(self):
        self.assertEqual(pf.get_pattern_size([[[[1, 1], [1, 1]], [[1, 1]]], [[[1, 1], [1, 1]], [[1, (1, 1)]]]]), 13)
        
    def test_make_indexed_pattern(self):
        pattern = pf.make_indexed_pattern(4, adding_code = [0,1,0,1], deleting_code = [0,0,1,0])
        self.assertEqual(pattern, [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [(10, 11), (12, 13)]], [[14], [(15, 16)]]]])
        
        pattern = pf.make_indexed_pattern(2, adding_code = [0,0], deleting_code = [1,0])
        self.assertEqual(pattern, [[0],[1]])
        
        pattern = pf.make_indexed_pattern(2, adding_code = [0,0], deleting_code = [1,0], starting_index = 1)
        self.assertEqual(pattern, [[1],[2]])
        
    def test_pattern_indexation(self):
        for i in range(2, 65):
            codes = pf.get_unique_codes(i)
            dim = round(math.log(i,2))
            for add, dele in codes:
                pattern = pf.make_polytope_pattern(dim, adding_code = add, deleting_code = dele)
                pattern_indexed_created = pf.make_indexed_pattern(dim, adding_code = add, deleting_code = dele)
                pattern_indexed_crafted = pf.index_this_pattern(pattern)
                self.assertEqual(pattern_indexed_created, pattern_indexed_crafted)
                
    def test_apply_chords_on_pattern(self):
        pattern = pf.make_indexed_pattern(3, adding_code = [], deleting_code = [])
        pat_one = pf.apply_chords_on_pattern(pattern, self.chord_sequence_global[:8])
        self.assertEqual(pat_one, [[['A', 'B'], ['C', 'D']], [['E', 'F'], ['G', 'A#']]])
        pattern = pf.make_polytope_pattern(3, adding_code = [], deleting_code = [])
        pat_two = pf.apply_chords_on_pattern(pattern, self.chord_sequence_global[:8])
        self.assertEqual(pat_one, pat_two)
        
    def test_extensive_apply_chords_on_pattern(self):
        for size in range(2, 65):
            codes = pf.get_unique_codes(size)
            dim = round(math.log(size,2))
            for add, dele in codes:
                pat_one = pf.make_indexed_pattern(dim, add, dele)
                pat_one = pf.apply_chords_on_pattern(pat_one, self.chord_sequence_global[:size])
                pat_two = pf.make_polytope_pattern(dim, add, dele)
                pat_two = pf.apply_chords_on_pattern(pat_two, self.chord_sequence_global[:size])
                self.assertEqual(pat_one, pat_two)

    def test_is_indexed_pattern(self):
        pattern = pf.make_polytope_pattern(3, adding_code = [], deleting_code = [])
        self.assertFalse(pf.is_indexed_pattern(pattern))
        pattern = pf.make_indexed_pattern(3, adding_code = [], deleting_code = [])
        self.assertTrue(pf.is_indexed_pattern(pattern))
        pattern[0][0][0] = 1
        with self.assertRaises(err.PatternToDebugError):
            pf.is_indexed_pattern(pattern)
            
    def test_flatten_nested_list(self):
        pattern = pf.make_indexed_pattern(5, adding_code = [1,1,0,0,0], deleting_code = [])
        self.assertEqual(pf.flatten_nested_list(pattern), [0, 1, 2, 3, 4, 5, 6, (7, 8), 9, 10, 11, 12, 13, 14, 15, (16, 17), 18, 19, 20, 21, 22, 23, 24, (25, 26), 27, 28, 29, 30, 31, 32, 33, (34, 35)])
        
    def test_flatten_pattern(self):
        pattern = pf.make_indexed_pattern(5, adding_code = [1,1,0,0,0], deleting_code = [])
        self.assertEqual(pf.flatten_pattern(pattern), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])
        

    def test_extract_pattern_from_indexed_pattern(self):
        pattern = pf.make_indexed_pattern(5, adding_code = [1,1,0,0,0], deleting_code = [])
        self.assertEqual(pf.extract_pattern_from_indexed_pattern(pattern), [[[[[1, 1], [1, 1]], [[1, 1], [1, (1, 1)]]],[[[1, 1], [1, 1]], [[1, 1], [1, (1, 1)]]]],
                                                                 [[[[1, 1], [1, 1]], [[1, 1], [1, (1, 1)]]],[[[1, 1], [1, 1]], [[1, 1], [1, (1, 1)]]]]])

if __name__ == '__main__':
    unittest.main()