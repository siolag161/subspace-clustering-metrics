import unittest as ut
from clustering.cluster2 import *
from clustering.evaluation import *


class TestEvaluator(ut.TestCase):

    def setUp(self):
        self.eva = SubspaceEvaluator()

    def test_entropy_pure(self):
        sc1 = SubspaceCluster([1,2,3,4], [1,2,3])
        sc2 = SubspaceCluster([1,2,3,4], [1,2,3])

        scs1 = SubspaceClustering([sc1])
        scs2 = SubspaceClustering([sc1, sc2])
        
        eva1 = SubspaceEvaluator(scs1)
        eva2 = SubspaceEvaluator(scs2)
        
        self.assertEqual(eva1.entropy_cluster(sc1), 0.0)
        self.assertEqual(eva2.entropy_cluster(sc1),0.0)

    def random_sample(self, li, nbr):
        sz = li.size
        rs = np.array([])
        for i in range(nbr):
            idx = np.random.randint(sz)
            rs = np.append(rs, li[idx])

        return rs

    def unique_random(self, nbr, sz):
        return np.unique(self.random_sample(np.arange(nbr), sz))

    def test_entropies(self):
        clusters1 = []
        for i in range(3):
            rs = self.unique_random(10, 3)
            sc = SubspaceCluster(rs, [1,2,3])
            clusters1.append(sc)

        
        scs1 = SubspaceClustering(clusters1)
        eva1 = SubspaceEvaluator(scs1)
        #self.assertEqual(eva1.entropy(scs1), 1.0)


    def test_entropy_pure_cluster(self):
        clusters = [[1,2,3], [ 4,5,6], [7,8,9,10]]

        subclusters = [SubspaceCluster(clust, clust) for clust in clusters]
        scs = SubspaceClustering(subclusters)
        eva1 = SubspaceEvaluator(scs)
        for clust in subclusters:                
            self.assertEqual(eva1.entropy_cluster(clust), 0.0)



    def test_rnia(self):
        cl_a1 = SubspaceCluster([1,2,3,4],[3,4])
        cl_a2 = SubspaceCluster([6,7],[4,5])
        cl_a3 = SubspaceCluster([4,5,6],[7,8,9])

        cl_b1 = SubspaceCluster([1,2],[3,4])
        cl_b2 = SubspaceCluster([3,4],[3,4])
        cl_b3 = SubspaceCluster([6,7],[5,6,7,8])

        cl_a = SubspaceClustering([cl_a1,cl_a2,cl_a3])
        cl_b = SubspaceClustering([cl_b1,cl_b2,cl_b3])

        eva = SubspaceEvaluator(cl_a)
        self.assertEqual(eva.rnia(cl_b), 13./25)


    def test_CE(self):
        cl_a1 = SubspaceCluster([1,2,3,4],[3,4])
        cl_a2 = SubspaceCluster([6,7],[4,5])
        cl_a3 = SubspaceCluster([4,5,6],[7,8,9])

        cl_b1 = SubspaceCluster([1,2],[3,4])
        cl_b2 = SubspaceCluster([3,4],[3,4])
        cl_b3 = SubspaceCluster([6,7],[5,6,7,8])

        cl_a = SubspaceClustering([cl_a1,cl_a2,cl_a3])
        cl_b = SubspaceClustering([cl_b1,cl_b2,cl_b3])

        eva = SubspaceEvaluator(cl_a)
        self.assertEqual(eva.CE(cl_b), 19./25)

    @staticmethod
    def dummy_cluster_quality(clust):
        return len(clust.objs)*len(clust.dims)
        

    def test_sorted(self):
        cl_a1 = SubspaceCluster([1,2,3,4],[3,4])
        cl_a2 = SubspaceCluster([6,7],[4,5])
        cl_a3 = SubspaceCluster([4,5,6],[7,8,9])

        cl_b1 = SubspaceCluster([1,2],[3,4])
        cl_b2 = SubspaceCluster([3,4],[3,4])
        cl_b3 = SubspaceCluster([6,7],[5,6,7,8])

        cl_a = SubspaceClustering([cl_a1,cl_a2,cl_a3])
        cl_b = SubspaceClustering([cl_b1,cl_b2,cl_b3])

        eva = SubspaceEvaluator(cl_a)
        #clc = eva.sorted(cl_a, lambda x:TestEvaluator.dummy_cluster_quality(x))
        #for cl in clc:
            #print cl.objs, cl.dims
        #self.assertEqual(eva.CE(cl_b), 33./25)  
