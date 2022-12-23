# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:38:53 2019

@author: amarmore
"""

import unittest
import polytopes.model.triad_manipulation as tm
import polytopes.triad_transformations as tt
import polytopes.model.errors as err
import polytopes.accelerated_polytopical_costs as acc_pc

import numpy as np

class AccelerationFunctionTests(unittest.TestCase):
            
    # def test_check_distance_function(self):
    #     circle_sharp =  ['C','Am','F','Dm','A#','Gm','D#','Cm','G#','Fm','C#','A#m','F#','D#m','B','G#m','E','C#m','A','F#m','D','Bm','G','Em']
    #     for i in range(24):
    #         chord_1 = circle_sharp[i]
    #         for j in range(24):
    #             chord_2 = circle_sharp[j]
    #             raw = tt.triadic_tonnetz_distance_symbol(chord_1, chord_2)
    #             accelerated = acc_pc.accelerated_triadic_tonnetz_distance_symbol(chord_1, chord_2)
    #             self.assertEqual(raw, accelerated)
    
    # def test_check_relation_function(self):
    #     circle_sharp =  ['C','Am','F','Dm','A#','Gm','D#','Cm','G#','Fm','C#','A#m','F#','D#m','B','G#m','E','C#m','A','F#m','D','Bm','G','Em']        
    #     relation_list = [0,'R','RL','RLR','LRPR','LRP','PR','P','PL','PLR','LPRP','LPRPR','PRPR','PRP','LPLR','LPL','LP','LPR','RP','RPR','LRLR','LRL','LR','L','LRPL','LPLP']
    #     for i in range(len(circle_sharp)):
    #         chord_1 = circle_sharp[i]
    #         for j in range(len(circle_sharp)):
    #             chord_2 = circle_sharp[j]
    #             accelerated = acc_pc.accelerated_triadic_tonnetz_relation_symbol(chord_1, chord_2)
    #             raw = tt.triadic_tonnetz_relation_symbol(chord_1, chord_2)
    #             if raw == 0:
    #                 self.assertEqual(raw, accelerated)
    #                 self.assertTrue(accelerated in relation_list)
    #             else:
    #                 iterator_str = ''
    #                 for rel in raw:
    #                     iterator_str += str(rel)
    #                 self.assertEqual(iterator_str, accelerated)
    #                 self.assertTrue(accelerated in relation_list)

                
    # def test_check_apply_relation_function(self):
    #     circle_sharp =  ['C','Am','F','Dm','A#','Gm','D#','Cm','G#','Fm','C#','A#m','F#','D#m','B','G#m','E','C#m','A','F#m','D','Bm','G','Em']
    #     relation_list = [0,'R','RL','RLR','LRPR','LRP','PR','P','PL','PLR','LPRP','LPRPR','PRPR','PRP','LPLR','LPL','LP','LPR','RP','RPR','LRLR','LRL','LR','L','LRPL','LPLP']
    #     for i in range(len(circle_sharp)):
    #         chord_1 = circle_sharp[i]
    #         for j in range(len(relation_list)):
    #             rel = relation_list[j]
    #             accelerated = acc_pc.accelerated_apply_triadic_tonnetz_relation_symbol(chord_1, rel)
    #             raw = tt.apply_triadic_tonnetz_relation_symbol(chord_1, rel)
    #             self.assertEqual(raw, accelerated)
                
    def test_check_get_chromatic_mvt_triads(self):
        circle_sharp =  ['C','Am','F','Dm','A#','Gm','D#','Cm','G#','Fm','C#','A#m','F#','D#m','B','G#m','E','C#m','A','F#m','D','Bm','G','Em']
        for i in range(len(circle_sharp)):
            chord_1 = circle_sharp[i]
            for j in range(len(circle_sharp)):
                chord_2 = circle_sharp[j]
                accelerated = acc_pc.accelerated_chromatic_mvt_triads(chord_1, chord_2)
                raw = tt.chromatic_mvt_triads(chord_1, chord_2)
                self.assertEqual(raw, accelerated)
                
    def test_check_get_voice_leading_transformation_symbol(self):
        circle_sharp =  ['C','Am','F','Dm','A#','Gm','D#','Cm','G#','Fm','C#','A#m','F#','D#m','B','G#m','E','C#m','A','F#m','D','Bm','G','Em']
        for i in range(len(circle_sharp)):
            chord_1 = circle_sharp[i]
            for j in range(len(circle_sharp)):
                chord_2 = circle_sharp[j]
                accelerated = acc_pc.accelerated_get_voice_leading_transformation_symbol(chord_1, chord_2)
                raw = tt.get_voice_leading_transformation_symbol(chord_1, chord_2)
                self.assertEqual(raw, accelerated)
        
if __name__ == '__main__':
    unittest.main()