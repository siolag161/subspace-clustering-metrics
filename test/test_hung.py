import unittest as ut

from clustering.hungarian import *
class TestHungarian(ut.TestCase):

    def test_hungarian(self):
        hg = Hungarian()
        a = np.array([[0, 1], [2, 3], [4, 5]])
        b = np.array([[0, 1, 0], [2, 3, 0], [4,5, 0]])
        self.assertTrue((hg._pad_to_square(a)== b).all())

        matrix = [[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]]

        hg._init_matrix(matrix)

        #self.assertTrue(hg.compute(matrix),0)
        self.assertEqual(hg._step_1()[1], 2)
        self.assertEqual(hg._step_2()[1], 3)
        self.assertEqual(hg._step_3()[1], 4)
        self.assertEqual(hg._step_4()[1], 6)
        self.assertEqual(hg._step_6()[1], 4)
        self.assertEqual(hg._step_4()[1], 5)
        self.assertEqual(hg._step_5()[1], 3)
        self.assertEqual(hg._step_3()[1], 4)      
        self.assertEqual(hg._step_4()[1], 6)       
        self.assertEqual(hg._step_6()[1], 4)    
        self.assertEqual(hg._step_4()[1], 6)    
        self.assertEqual(hg._step_6()[1], 4)    
        self.assertEqual(hg._step_4()[1], 5)
        self.assertEqual(hg._step_5()[1], 3)     
        self.assertEqual(hg._step_3()[1], 7)  
        

    def test_cost_matrix(self):
        hg = Hungarian()

        c1 = [1, 1, 2, 3, 3, 4, 4, 4, 2]
        c2 = [2, 2, 3, 1, 1, 4, 4, 4, 3]

        mt = hg.make_cost_matrix(c1,c2)
        rs = hg.compute(mt)
        self.assertNotEqual(rs, None)
