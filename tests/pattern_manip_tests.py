# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:03:20 2019

@author: amarmore
"""

import unittest
import polytopes.pattern_factory as pf
import polytopes.pattern_manip as pm
import polytopes.segmentation_helper as sh
import polytopes.model.errors as err
from polytopes.model.chord import Chord
import math
import numpy as np

class PatternManipulationTests(unittest.TestCase):
    
    def test_indexing_on_examples(self):
        pattern = pf.make_indexed_pattern(4, adding_code = [0,1,0,1], deleting_code = [0,0,1,0])
        # pattern = [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [(10, 11), (12, 13)]], [[14], [(15, 16)]]]] 
        self.assertEqual(pm.get_index_from_element(2, pattern), [0, 0, 1, 0])
        self.assertEqual(pm.get_index_from_element(6, pattern), [0, 1, 1, 0])
        self.assertEqual(pm.get_index_from_element(13, pattern), [1,0,1,(1,1)])
        self.assertEqual(pm.get_index_from_element(13, pattern), [1, 1, 0, 0])
        
    def test_indexing_on_examples(self):
        pattern = pf.make_indexed_pattern(4, adding_code = [0,1,0,1], deleting_code = [0,0,1,0])
        # pattern = [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [(10, 11), (12, 13)]], [[14], [(15, 16)]]]] 
        self.assertEqual(pm.get_element_from_index([0, 0, 1, 1], pattern), 3)
        self.assertEqual(pm.get_element_from_index([0, 1, 0, 0], pattern), 4)
        self.assertIsNone(pm.get_element_from_index([0,0,1,(1,1)], pattern))
        self.assertEqual(pm.get_element_from_index([1, 1, 1, (0, 0)], pattern), 15)

    def test_indexing_and_retrieving_from_index(self):
        for size in range(2, 70):
            dim = round(math.log(size,2))
            for add, dele in pf.get_codes(size):
                pattern = pf.make_indexed_pattern(dim, adding_code = add, deleting_code = dele, starting_index = 0)
                for i in range(0,pf.get_pattern_size(pattern)):
                    idx_elt = pm.get_index_from_element(i, pattern)
                    self.assertEqual(i, pm.get_element_from_index(idx_elt, pattern))
                    
    def test_delete_tuples(self):
        self.assertEqual(pm.delete_tuples([0, 1, 0, 0]), [0, 1, 0, 0])
        self.assertEqual(pm.delete_tuples([0, 1, 0, (0,1)]), [0, 1, 0, 0])
        self.assertEqual(pm.delete_tuples([0, 1, 0, (0,0)]), [0, 1, 0, 0])
        self.assertEqual(pm.delete_tuples([0, 1, 0, (1,0)]), [0, 1, 0, 1])
        self.assertEqual(pm.delete_tuples([0, 1, 0, (1,1)]), [0, 1, 0, 1])
        
    def test_add_indexes(self):
        self.assertEqual(pm.add_indexes([1,0,0], [0,0,1]), [1,0,1])
        self.assertEqual(pm.add_indexes([1,0,0], [0,0,(0,1)]), [1,0,(0,1)])
        self.assertEqual(pm.add_indexes([1,0,(1,0)], [0,0,(0,1)]), [1,0,(1,1)])
        with self.assertRaises(err.InvalidIndexSizeException):
            pm.add_indexes([1,0,0], [0,0,1,0])
        with self.assertRaises(err.InvalidIndexException):
            pm.add_indexes([1,0,0], [1,1,0])
        
    
    def test_some_antecedents_ground_truth(self):
        pattern = pf.make_indexed_pattern(4, adding_code = [0,1,0,1], deleting_code = [0,0,1,0])
        # pattern = [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [(10, 11), (12, 13)]], [[14], [(15, 16)]]]]
        elt_idx = pm.get_index_from_element(5, pattern)
        for ant in pm.get_antecedents_from_index(elt_idx, pattern):
            self.assertTrue(ant in [1,4])
            
        elt_idx = pm.get_index_from_element(10, pattern)
        for ant in pm.get_antecedents_from_index(elt_idx, pattern):
            self.assertTrue(ant in [2,8])
            
        elt_idx = pm.get_index_from_element(13, pattern)
        for ant in pm.get_antecedents_from_index(elt_idx, pattern):
            self.assertTrue(ant in [11,12])
                    
    def test_no_antecedent_is_none(self):
        for size in range(2, 70):
            dim = round(math.log(size,2))
            for add, dele in pf.get_codes(size):
                pattern = pf.make_indexed_pattern(dim, adding_code = add, deleting_code = dele, starting_index = 0)
                for i in range(1,pf.get_pattern_size(pattern)):
                    idx_elt = pm.get_index_from_element(i, pattern)
                    ant = pm.get_antecedents_from_index(idx_elt, pattern)
                    self.assertNotEqual(ant, None)
                    
    def test_some_pivot_ground_truth(self):
        pattern = pf.make_indexed_pattern(4, adding_code = [0,1,0,1], deleting_code = [0,0,1,0])
        # pattern = [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [(10, 11), (12, 13)]], [[14], [(15, 16)]]]]
        elt_idx = pm.get_index_from_element(5, pattern)
        ant_idx = pm.get_index_from_element(1, pattern)
        self.assertEqual(pm.get_pivot_index_from_index(elt_idx, ant_idx), [0, 1, 0, 0])
            
        elt_idx = pm.get_index_from_element(10, pattern)
        ant_idx = pm.get_index_from_element(2, pattern)
        self.assertEqual(pm.get_pivot_index_from_index(elt_idx, ant_idx), [1, 0, 0, 0])
            
        elt_idx = pm.get_index_from_element(13, pattern)
        ant_idx = pm.get_index_from_element(12, pattern)
        self.assertEqual(pm.get_pivot_index_from_index(elt_idx, ant_idx), [0, 0, 0, 0])
                    
    def test_no_pivot_is_none(self):
        for size in range(2, 70):
            dim = round(math.log(size,2))
            for add, dele in pf.get_codes(size):
                pattern = pf.make_indexed_pattern(dim, adding_code = add, deleting_code = dele, starting_index = 0)
                for i in range(1,pf.get_pattern_size(pattern)):
                    idx_elt = pm.get_index_from_element(i, pattern)
                    ants_idx = pm.get_antecedents_index_from_index(idx_elt)
                    for ant_idx in ants_idx:
                        self.assertNotEqual(pm.get_pivot_index_from_index(idx_elt, ant_idx), None)
            
    def test_some_successors_ground_truth(self):
        pattern = pf.make_indexed_pattern(4, adding_code = [0,1,0,1], deleting_code = [0,0,1,0])
        # pattern = [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [(10, 11), (12, 13)]], [[14], [(15, 16)]]]]
        elt_idx = pm.get_index_from_element(5, pattern)
        self.assertEqual(pm.get_successors_from_index(elt_idx, pattern), [7])

            
        elt_idx = pm.get_index_from_element(10, pattern)
        for suc in pm.get_successors_from_index(elt_idx, pattern):
            self.assertTrue(suc in [11,12,15])
            
        elt_idx = pm.get_index_from_element(13, pattern)
        self.assertEqual(pm.get_successors_from_index(elt_idx, pattern), [])
         
    def test_no_error_when_computing_successors(self):
        for size in range(2, 70):
            dim = round(math.log(size,2))
            for add, dele in pf.get_codes(size):
                pattern = pf.make_indexed_pattern(dim, adding_code = add, deleting_code = dele, starting_index = 0)
                for i in range(0,pf.get_pattern_size(pattern)):
                    idx_elt = pm.get_index_from_element(i, pattern)
                    suc = pm.get_successors_from_index(idx_elt, pattern)
                    
                    
    def test_successors_and_antecedents_are_equivalent(self):
        for size in range(4, 70):
            for patt, _, _, antecedents, successors, _ in sh.compute_patterns_with_antecedents_for_size(size):
                for elt in range(1, size):
                    for suc in successors[elt]:
                        self.assertTrue(elt in np.array(antecedents[suc])[:,0]) # Checking that elt is in the antecedents of its successors
                    for ant in np.array(antecedents[elt])[:,0]:
                        if ant != 0: # Successors of 0 are None, as they're not considered valid in Guichaoua's paradigm.
                            self.assertTrue(elt in successors[ant]) # Checking that elt is a successors of its antecedents
    
    def test_global_successors_and_antecedents_are_equivalent(self):
        for size in range(4, 70):
            for patt, _, _, global_antecedents, global_successors, _ in sh.compute_patterns_with_global_antecedents_for_size(size):
                for elt in range(1, size):
                    for suc in global_successors[elt]:
                        self.assertTrue((elt in np.array(global_antecedents[suc])[:,0]) or (elt in np.array(global_antecedents[suc])[:,1])) # Checking that elt is in the antecedents of its successors
                    for ant in np.array(global_antecedents[elt])[:,0]:
                        if ant != 0: # Successors of 0 are None, as they're not considered valid in Guichaoua's paradigm.
                            self.assertTrue(elt in global_successors[ant]) # Checking that elt is a successors of its antecedents   
                            
    def test_find_primers_of_low_level_systems(self):
        pattern = [[[0, 1], [2, 3]], [[4, (5, 6)], [7, (8, 9)]]]
        self.assertEqual(pm.find_primers_of_low_level_systems(pattern), [0, 4])
        
        pattern = pf.make_indexed_pattern(5, adding_code = [1,0,0,1,1], deleting_code = [])
        #pattern = [[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11]], [[(12, 13), (14, 15)], 
        #           [(16, 17), (18, 19)]]]], [[[[20, 21], [22, 23]], [[24, 25], [26, 27]]], [[[28, 29], [30, 31]], [[(32, 33), (34, 35)], [(36, 37), (38, 39)]]]]]
        self.assertEqual(pm.find_primers_of_low_level_systems(pattern), [[[0, 4], [8, 12]], [[20, 24], [28, 32]]])

    def test_get_under_primers(self):
        pattern = pf.make_indexed_pattern(5, adding_code = [1,0,0,1,1], deleting_code = [])
        self.assertEqual(pm.get_under_primers(pattern),[[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    
    def test_ppp_example(self):
        pattern = [[[0, 1], [2, 3]], [[4, (5, 6)], [7, (8, 9)]]]
        for a_ppp in pm.generate_ppp(pattern):
            self.assertTrue(a_ppp in [[[[0, 1], [2, 3]], [[4, (5, 6)], [7, (8, 9)]]],[[[0, 1], [4, (5, 6)]], [[2, 3], [7, (8, 9)]]],[[[0, 2], [4, 7]], [[1, 3], [(5, 6), (8, 9)]]]])

        pattern = [[[0, 1], [2, (3, 4)]], [[5, 6], [7, (8, 9)]]]
        for a_ppp in pm.generate_ppp(pattern):
            self.assertTrue(a_ppp in [[[[0, 1], [2, (3, 4)]], [[5, 6], [7, (8, 9)]]],[[[0, 1], [5, 6]], [[2, (3, 4)], [7, (8, 9)]]],[[[0, 2], [5, 7]], [[1, (3, 4)], [6, (8, 9)]]]])
        
        pattern = [[[0, 1], [2, 3]], [[4, 5], [(6, 7), (8, 9)]]]
        for a_ppp in pm.generate_ppp(pattern):
            self.assertTrue(a_ppp in [[[[0, 1], [2, 3]], [[4, 5], [(6, 7), (8, 9)]]],
                                      [[[0, 1], [4, 5]], [[2, 3], [(6, 7), (8, 9)]]],
                                      [[[0, 2], [4, (6, 7)]], [[1, 3], [5, (8, 9)]]]])
            
            self.assertFalse(a_ppp in [[[[0, 1], [2, (3, 4)]], [[5, 6], [7, (8, 9)]]],
                                       [[[0, 1], [5, 6]], [[2, (3, 4)], [7, (8, 9)]]],
                                       [[[0, 2], [5, 7]], [[1, (3, 4)], [6, (8, 9)]]]])
            
        pattern = [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]]
        for a_ppp in pm.generate_ppp(pattern):
            self.assertTrue(a_ppp in [[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],[[[8, 9], [10, 11]], [[12, 13], [14, 15]]]],
                                      [[[[0, 1], [4, 5]], [[2, 3], [6, 7]]],[[[8, 9], [12, 13]], [[10, 11], [14, 15]]]],
                                      [[[[0, 1], [8, 9]], [[2, 3], [10, 11]]],[[[4, 5], [12, 13]], [[6, 7], [14, 15]]]],
                                      [[[[0, 2], [4, 6]], [[1, 3], [5, 7]]],[[[8, 10], [12, 14]], [[9, 11], [13, 15]]]],
                                      [[[[0, 2], [8, 10]], [[1, 3], [9, 11]]],[[[4, 6], [12, 14]], [[5, 7], [13, 15]]]],
                                      [[[[0, 4], [8, 12]], [[1, 5], [9, 13]]],[[[2, 6], [10, 14]], [[3, 7], [11, 15]]]]])


    def test_direct_antecedents_on_ppp(self):
        pattern = [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]]
        self.assertEqual(pm.compute_direct_antecedents(pattern),[None, 0, 0, (0, 2, 1), 0, 4, 4, (4, 6, 5), 0, 8, 8, (8, 10, 9), (0, 8, 4), 12, 12, (12, 14, 13)])
        pattern = [[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10, 11], [14, 15]]]]
        self.assertEqual(pm.compute_direct_antecedents(pattern),[None, 0, 0, 2, 0, (0, 4, 1), 2, (2, 6, 3), 0, 8, (0, 8, 2), 10, 8, (8, 12, 9), 10, (10, 14, 11)])
        pattern = [[[[0, 1], [8, 9]], [[2, 3], [10, 11]]], [[[4, 5], [12, 13]], [[6, 7], [14, 15]]]]
        self.assertEqual(pm.compute_direct_antecedents(pattern),[None, 0, 0, 2, 0, 4, (0, 4, 2), 6, 0, (0, 8, 1), 2, (2, 10, 3), 4, (4, 12, 5), 6, (6, 14, 7)])
        pattern = [[[[0, 2], [4, 6]], [[1, 3], [5, 7]]], [[[8, 10], [12, 14]], [[9, 11], [13, 15]]]]
        self.assertEqual(pm.compute_direct_antecedents(pattern),[None, 0, 0, 1, 0, 1, (0, 4, 2), (1, 5, 3), 0, (0, 8, 1), 8, 9, 8, 9, (8, 12, 10), (9, 13, 11)])
        pattern = [[[[0, 2], [8, 10]], [[1, 3], [9, 11]]], [[[4, 6], [12, 14]], [[5, 7], [13, 15]]]]
        self.assertEqual(pm.compute_direct_antecedents(pattern),[None, 0, 0, 1, 0, (0, 4, 1), 4, 5, 0, 1, (0, 8, 2), (1, 9, 3), 4, 5, (4, 12, 6), (5, 13, 7)])                
 
    def test_swap_chord_sequence(self):
        pattern = [[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], [[[8, 9], [12, 13]], [[10, 11], [14, 15]]]]
        chord_sequence = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A#', 'C#', 'D#', 'F#', 'G#', 'A', 'B','C', 'D']
        self.assertEqual(pm.swap_chord_sequence(chord_sequence, pf.flatten_pattern(pattern)), ['A','B', 'E', 'F', 'C', 'D', 'G', 'A#', 'C#', 'D#', 'A', 'B', 'F#', 'G#', 'C', 'D'])

if __name__ == '__main__':
    unittest.main()