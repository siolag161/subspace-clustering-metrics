import numpy as np

class SubspaceCluster:
    
    @staticmethod
    def cardinal_interesting(clust, alpha = 1, beta = 1):
        return alpha*np.log(len(clust.objs))+ beta*np.log(len(clust.dims))
    
    def __init__(self, objs, dims, **kwargs):
        self.objs = np.unique(objs)
        self.dims = np.unique(dims)
        self.info = kwargs       
 

class SubspaceClustering:
    def __init__(self, clusters = []):
        self.clusters = clusters
               
    def size(self):
        return len(self.clusters) 

    def add(self, clust):
        self.clusters.append(clust)

    def __iter__(self):
        return iter(self.clusters)

############# SOME HELPER CLASSES & METHODS ################

class ClusteringReader:
    
    def __init__(self, fname, header=True):
        self.has_header = header
        self.path = fname
        self.clts = {}

    def set_path(self, fname):
        self.path = fname
        
    def load(self):
        
        import os, csv

        self.clts.clear()
        if os.path.isfile(self.path):
            with open(self.path, 'rb') as fi:           
                rd = csv.DictReader(fi)              

                for row in rd:
                    algo = row['algorithm']
                    params = row['parameters']
                    objs = row['objects']
                    dims = row['dimensions']
                    run = row['run']

                    clustering_id = '%s-%s' %(algo, run)
                    #print clustering_id
                    self.clts.setdefault(clustering_id, SubspaceClustering() )

                    dimensions = [int(dim) for dim in dims.split(',')]
                    for obs in objs.split(';'):
                        objects = [int(obj) for obj in obs.split(',')]                                                          
                        clust = SubspaceCluster(objects, dimensions)
                        self.clts[clustering_id].add(clust)
        
        return self.clts
