import unittest as ut
from clustering.cluster2 import *
from clustering.evaluation import *


class TestLoading(ut.TestCase):
    def setUp(self):
        self.path = "clique.csv"



    def test_dominance(self):
        
        rd = ClusteringReader(self.path)

        clts = rd.load()
        self.assertTrue(len(clts)>0)
        self.assertEqual(len(clts), 1)
        k=clts.keys()[0]
        self.assertEqual(clts[k].size(), 7)
        dims = set([])
        for clust in clts[k]:
            #dims.add(clust.dims)
            dim = ','.join(str(d) for d in clust.dims)
            dims.add(dim)

        self.assertEqual(len(dims), 7)
        #eva = SubspaceEvaluator(clts[k])


        non_dominated = SubspaceEvaluator.non_dominated_clustering(clts[k], lambda x: cardinal_interesting(x))
        for clu in non_dominated:
            print clu            
        self.assertEqual(0, 1)


def cardinal_interesting(clu, alpha = 1, beta = 1):
    import math
    return alpha*math.log(len(clu.objs))+ beta*math.log(len(clu.dims))

