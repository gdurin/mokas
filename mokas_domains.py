import pickle
import numpy as np
import mahotas
import getLogDistributions as gLD
import matplotlib.pyplot as plt
import itertools
import scipy.spatial as spatial
from skimage.morphology import remove_small_holes
from skimage.draw import line_aa

class Domains:
    """
    This class calculates various properties of a domain,
    given a map of the switch times as a 2D array (switch2D)
    """
    def __init__(self, switch2D, no_switch_value=-1, NNstructure=None):
        if not NNstructure:
            self.NNstructure = np.ones((3, 3))
        else:
            self.NNstructure = NNstructure
        sw = np.unique(switch2D)
        self.sw = sw[sw != no_switch_value]
        self.switch2D = switch2D
        self.switched_domain = switch2D >= self.sw[0]
        self.is_single_domain, self.initial_clusters, self.n_initial_clusters \
                = self._is_single_domain(self.switched_domain, self.NNstructure)
        self.max_switch = self.max_switch_not_touching_edges()

    def _is_single_domain(self, domain, NNstructure):
        im, n_cluster = mahotas.label(domain, NNstructure)
        if n_cluster > 1:
            print("There are %d clusters" % n_cluster)
            return False, im, n_cluster
        else:
            print("It is a single domain")
            return True, im, 1

    def get_initial_domain(self, is_remove_small_holes=True):
        """
        The detection of the initial domain has a lot of problems
        We need to differenciate between:
        1. a large domain with a few small clusters
        2. similar domains not connected
        """
        if self.is_single_domain:
            q = np.copy(self.switched_domain)
        else:
            n = self.n_initial_clusters + 1
            im = np.copy(self.initial_clusters)
            for i, j in itertools.combinations(range(1, n), 2):
                im_i = im == i
                im_j = im == j
                p_i = mahotas.labeled.bwperim(im_i)
                p_j = mahotas.labeled.bwperim(im_j)
                indices_i = np.asarray(np.nonzero(p_i)).T
                indices_j = np.asarray(np.nonzero(p_j)).T
                mytree = spatial.cKDTree(indices_i)
                dist, indexes = mytree.query(indices_j)
                imin = np.argmin(dist)
                q_i = mytree.data[indexes[imin]]
                mytree = spatial.cKDTree(indices_j)
                dist, indexes = mytree.query(indices_i)
                imin = np.argmin(dist)
                q_j = mytree.data[indexes[imin]]
                print(q_i, q_j)
                x_i, y_i = q_i.astype(int)
                x_j, y_j = q_j.astype(int)
                # Add a line between the two points
                rr, cc, val = line_aa(x_i, y_i, x_j, y_j)
                im[rr, cc] = 1
            q = im > 0
        if is_remove_small_holes:
            q = remove_small_holes(q)
        im, n_cluster = mahotas.label(~q, self.NNstructure)
        im = mahotas.labeled.remove_bordering(im)
        if n_cluster > 1:
            print("There are more clusters than expected: I take the largest")
            im, n_cluster = mahotas.labeled.relabel(im)
            size_clusters = mahotas.labeled.labeled_size(im)[1:]
            index = size_clusters.argmax()
            #size_central_domain = size_clusters[index]
            im = im == index + 1
        return im

    def max_switch_not_touching_edges(self):
        """
        Calculated the max switch with a fully internal domain 
        It is used to calculate the initial nucleated domain 
        """
        q = np.copy(self.switch2D)
        rows, cols = q.shape
        for switch in self.sw[::-1]:
            dm = (q <= switch) & (q != -1)
            min_row, max_row, min_col, max_col = mahotas.bbox(dm)
            if (min_row == 0) or (min_col==0) or (max_row==rows) or (max_col==cols):
                next
            else:
                return switch

if __name__ == "__main__":
    filename = "switch2D_05.pkl"
    with open(filename, 'rb') as f:
        switch2D = pickle.load(f)
    domain = Domains(switch2D)
    plt.figure()
    plt.imshow(domain.initial_clusters, interpolation='None')
    plt.figure()
    initial_domain = domain.get_initial_domain()
    plt.imshow(initial_domain, interpolation='None')
    plt.show()
