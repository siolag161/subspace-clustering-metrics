
import re
import csv

from cluster import SubspaceCluster

class ClusteringResultProcessor:
    def __init__(self, path,  **kwargs):
        self.path = path

    def dim_list_to_string(self, dims):
        ds = sorted(dims)
        dim_str = [str(d) for d in ds]
        dim_str = "-".join(dim_str)
        
        return dim_str

    def read(self):
        rs = {}
        with open(self.path, 'r') as f:            
            reader = csv.reader(f)
            for row in reader:
                algo = row[0]
                dims = row[1]
                objs = row[2]
                ite  = row[3]  

                dims = dims.split(',')
                obj_set = objs.split(';')

                for obj in obj_set:
                    obj = obj.split(',')   
                    clust = SubspaceCluster(dims, obj, algo+"-"+ite)
                    rs.setdefault(clust, 0)
        
        return rs

    @staticmethod
    def more_cardinal_interesting(clust1, clust2, alpha = 0.5, beta = 0.5):
        return clust1.cardinal_interestingness(alpha, beta) > clust2.cardinal_interestingness(alpha, beta) 

    @staticmethod
    def more_contiguous_interesting(clust1, clust2):
        pass
    
    def dominated_list(self, clusters, more_interesting):
        dominated_list = dict([(clust,0) for clust in clusters])
        for clust1 in clusters:
            for clust2 in clusters:
                if clust1.is_similar(clust2) and more_interesting(clust2, clust1):
                    dominated_list[clust1] = 1
                    break
        return dominated_list
                    
