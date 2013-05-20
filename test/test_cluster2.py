import unittest as ut
from clustering.cluster2 import *
from clustering.evalutation import *

class TestCluster2(ut.TestCase):

    def setUp(self):
        self.eva = SubspaceEvaluator()
    
    def test_precision_equal(self):
        sc1 = SubspaceCluster([1,2,3,4], [1,2,3])
        sc2 = SubspaceCluster([1,2,3,4], [1,2,3])
        
        self.assertEqual(self.eva.precision_objs(sc1, sc2), 1)
        self.assertEqual(self.eva.precision_dims(sc1, sc2), 1)

    def test_precision_subsets(self):
        sc1 = SubspaceCluster([1,2,3,4], [1,2,3])
        sc2 = SubspaceCluster([3,4], [1,2,3,4,5,6,7,8])
        
        self.assertEqual(self.eva.precision_objs(sc1, sc2), 0.5)
        self.assertEqual(self.eva.precision_objs(sc2, sc2), 1)

        self.assertEqual(self.eva.precision_dims(sc1, sc2), 1)
        self.assertEqual(self.eva.precision_dims(sc2, sc1), 3./8)

    def test_similarity(self):
        sc1 = SubspaceCluster([1,2,3,4], [1,2,3])
        sc2 = SubspaceCluster([3,4], [1,2,3,4,5,6,7,8])

        self.assertTrue(self.eva.similar_to(sc1, sc2, 0.5, 1))
        self.assertFalse(self.eva.similar_to(sc1, sc2, 0.7, 1))

        self.assertTrue(self.eva.similar_to(sc2, sc1, 1, 3.0/8))
        self.assertFalse(self.eva.similar_to(sc2, sc1, 1, 3.0/7))


        
    def test_dominance(self):

        sc1 = SubspaceCluster([1,2,3,4], [1,2,3])
        sc2 = SubspaceCluster([3,4], [1,2,3,4,5,6,7,8])

        self.assertTrue(self.eva.similar_to(sc1, sc2, 0.5, 1))
        self.assertTrue(self.eva.dominated_by(sc1, sc2, SubspaceCluster.cardinal_interesting))

        self.assertTrue(self.eva.similar_to(sc2, sc1, 1, 3.0/8))
       
