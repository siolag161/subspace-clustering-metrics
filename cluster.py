import math

class Cluster:
    def __init__(self, objects):
        pass


class SubspaceCluster:
    def __init__(self, objs, dims, info):
        self.objs = set(objs)
        self.dims = set(dims)
        self.info = info

    def similarity(self, clust):
        ob_it = set.intersection(self.objs, clust.objs)
        dim_it = set.intersection(self.dims, clust.dims)
        
        sim_obs = len(ob_it)*1.0/len(self.objs)
        sim_dim = len(dim_it)*1.0/len(self.dims)

        return [sim_obs, sim_dim]

    def is_similar(self, clust, obj_thres = 0.5, dim_thres = 0.5):
        [sim_obs, sim_dim] = self.similarity(clust)
        return (sim_obs > obj_thres and sim_dim > dim_thres)

    def is_inferior(self, clust, quality_funct):
        """
        return true if a < b w.r.t the comparator cmp <-> cmp(self, clust) == -1
        """
        return quality_funct(self) < quality_func(clust)

    def is_dominated(self, clust, func):
        return self.similar(clust) and self.is_inferior(clust, funct)

    def cardinal_interestingness(self, alpha = 1, beta = 1):
        return alpha*math.log(len(self.objs))+ beta*math.log(len(self.dims))



class SubspaceClustering:

    def __init__(self, full_space, clusters, info):
        self.full_space = full_space
        self.clusters = clusters
        self.info = info

    def _cluster_labels(self):
        rs = dict([(obj[0], 0) for obj in full_space])
        label = 1
        for cluster in clusters:
            for obj in cluster.objs:
                rs[obj] = label
            label += 1
        return rs
        
    def contiguity_scores(self):
        labels = self.cluster_labels()             
        
        count = 1
        for i in range(1, len(self.full_space)):
            if labels[self.full_space[i][0]] != labels[self.full_space[i-1][0]]:
                count += 1

        # now we have to normalize it
        s = sum([len(clust)**2 for clust in self.clusters])
        n = len(self.full_space)
        expected_val = (n**2-s)/(n)           
        return count/expected_val 
        
