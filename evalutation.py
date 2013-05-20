import numpy as np
from clustering.cluster2 import *


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
                print etp
                etp += -proba*np.log(proba)
                
        return etp



     #########################
        


