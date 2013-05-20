import numpy as np
from clustering.cluster2 import *
from clustering.hungarian import Hungarian as Hg

class SubspaceEvaluator:
    """ 
    based on the paper 
    Evaluating Clustering in Subspace Projections of High Dimensional Data
    """
    def __init__(self, clustering = None):
        self.set_reference(clustering)
        
    def set_reference(self, clustering):
        self.ref = clustering    
                   
    def f1(self, clust):
        """ compute the f1-measure for clustering, it will use the f1 for cluster"""
        if not clust.clusters or not self.ref.clusters:
            return 0
        sz = self.ref.size()
        
        f_1 = 0.0
        for cli in self.ref.clusters:
            clq = [cli.f1(clj) for clj in clust.clusters]
            clq_max = max(clq)
            f_1 += clq_max
        return f1/sz

    ################### STATIC METHODS ###############
    @staticmethod
    def f1_cluster(clust1, clust2):
        recall = SubspaceEvaluator.recall_objs(clust1, clust2)
        precision = SubspaceEvaluator.precision_objs(clust1, clust2)
        if (recall+precision != 0):
            return (2*recall*precision)/(recall+precision)
        else:
            return 0
        
    @staticmethod
    def similarity(clust1, clust2):
        return (SubspaceEvaluator.precision_objs(clust1, clust2), 
                SubspaceEvaluator.precision_dims(clust1, clust2))

    @staticmethod
    def similar_to(clust1, clust2, objs_threshold = 0.5, dims_threshold = 0.5):
        sim_obj, sim_dim = SubspaceEvaluator.similarity(clust1, clust2)
        return sim_obj >= objs_threshold and sim_dim >= dims_threshold

    @staticmethod
    def inferior_to(clust1, clust2, func):     
        return func(clust1)<func(clust2)     
    
    @staticmethod
    def dominated_by(clust1, clust2, func, objs_threshold = 0.5, dims_threshold = 0.5):
        return SubspaceEvaluator.similar_to(clust1, clust2, objs_threshold, dims_threshold) and \
          SubspaceEvaluator.inferior_to(clust1, clust2, func)


    ####################PRECISION & RECALL######################
    @staticmethod
    def precision_objs(clust1, clust2):
        intersection = np.intersect1d(clust1.objs, clust2.objs)
        if (clust1.objs.size != 0):
            return intersection.size*1.0/clust1.objs.size
        else:
            return 0

    @staticmethod
    def precision_dims(clust1, clust2):
        intersection = np.intersect1d(clust1.dims, clust2.dims)
        if (clust1.dims.size != 0):
            return intersection.size*1.0/clust1.dims.size
        else:
            return 0
        
    @staticmethod    
    def recall_dims(clust1, clust2):
        return SubspaceEvaluator.precision_dims(clust2, clust1)
    
    @staticmethod
    def recall_objs(clust1, clust2):
        return SubspaceEvaluator.precision_objs(clust2, clust1)

    ###################### ENTROPY ################    
    def entropy(self, clustering):
        """ compute the entropy-measure of clustering """

        entropy_max = max([self.entropy_cluster(p) for p in clustering.clusters])
        entropy_max_sum = entropy_max*sum(p.objs.size for p in clustering.clusters)      

        if (entropy_max_sum == 0):
            return 1
        else:
            entropy_weighted_sum = sum([p.objs.size*self.entropy_cluster(p) for 
                         p in clustering.clusters])
            entropy_weighted_average = entropy_weighted_sum/entropy_max_sum;
            return 1-entropy_weighted_average        
    
        
        
    def entropy_cluster(self, cluster):
        etp = 0.0

        for clust in self.ref.clusters:
            proba = SubspaceEvaluator.precision_objs(cluster, clust)            
            if (proba != 0):
                etp += -proba*np.log(proba)
                
        return etp

    #########################

    def matrix_intersection(self, A, B):
        nrows = max(A.shape[0], B.shape[0])
        ncols = max(A.shape[1], B.shape[1])

    def make_point_tuple(self, clustering):
        clusters = clustering.clusters
        rs = set([])

        for cluster in clusters:
            for row in cluster.objs:
                for col in cluster.dims:
                    rs.add((row, col))
                    
        return rs

    
    def rnia(self, clustering):
        
        s1 = self.make_point_tuple(self.ref)
        s2 = self.make_point_tuple(clustering)

        U = len(set.union(s1, s2))
        
        if U==0:
            return 0        
        I = len(set.intersection(s1, s2))    
     
        return (U-I)*1.0/U

     #########################   

    def cluster_intersection(self, clustA, clustB):
        rsA = set([])
        for row in clustA.objs:
                for col in clustA.dims:
                    rsA.add((row, col))

        rsB = set([])
        for row in clustB.objs:
                for col in clustB.dims:
                    rsB.add((row, col))

        intersection = rsA.intersection(rsB)

        return len(intersection)
    
    def make_ce_matrix(self, clustering):
        clt = clustering

        row = self.ref.size()
        col = clt.size()
        mat = 0*np.ones([row,col], dtype=int)
       
        for i in range(row):
            for j in range(col):
                clustA = self.ref.clusters[i]
                clustB = clt.clusters[j]
                mat[i,j] = self.cluster_intersection(clustA, clustB)

        return mat


    def clustering_error(self, clustering):
        mat = self.make_ce_matrix(clustering)

        hung_solver = Hg()
        rs, D_Max = hung_solver.compute(mat, True)
        
        s1 = self.make_point_tuple(self.ref)
        s2 = self.make_point_tuple(clustering)

        U = len(set.union(s1, s2))
        
        if U==0:
            return 0        
        return (U-D_Max)/(U)

    
    def CE(self, clustering):
        return self.clustering_error(clustering)


    ############# SORT AND FILTER NON-DOMINATED ############

    @staticmethod
    def sorted_clusters(clusters, func, reverse=True):
        """ sort the clusters of the clustering passed in argument"""
        return sorted(clusters, key=func, reverse=reverse)
    
    @staticmethod
    def sorted_clustering(clustering, func, reverse=True):
        """ sort the clusters of the clustering passed in argument"""
        return sorted_clusters(clustering.clusters, func, reverse)


    @staticmethod
    def non_dominated_clustering(clustering, func):
        """ get the non-dominated ones from clustering"""  
        return SubspaceEvaluator.non_dominated_clusters(clustering.clusters, func)


    @staticmethod
    def non_dominated_clusters(clusters, func):
        """ TODO: sort first and then select --> n*ln(n) """
        sz = len(clusters)
        idx = 0*np.ones([sz])
        for i in range(sz):
            clA = clusters[i]
            for j in range(sz):
                clB = clusters[j]
                if SubspaceEvaluator.dominated_by(clA, clB, func):
                    idx[i] = 1
                if SubspaceEvaluator.dominated_by(clB, clA, func):
                    idx[j] = 1

        return np.nonzero(idx==0)[0]
                               
                       
