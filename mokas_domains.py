import pickle
import numpy as np
import mahotas
import getLogDistributions as gLD
import matplotlib.pyplot as plt
import itertools
import scipy.spatial as spatial
import scipy.ndimage as nd
from skimage.morphology import remove_small_holes
from skimage.draw import line_aa, line
import matplotlib.pyplot as plt
import sys

SPACE = "###"
NNstructure = np.array([[0,1,0],[1,1,1],[0,1,0]])


class Domains:
    """
    This class calculates various properties of a domain,
    given a map of the switch times as a 2D array (switch2D)
    """
    def __init__(self, switch2D, no_switch_value=-1, NNstructure=None,
                    is_remove_small_holes=True, small_holes_area=64):
        if not NNstructure:
            self.NNstructure = np.ones((3, 3))
        else:
            self.NNstructure = NNstructure
        sw = np.unique(switch2D)
        self.sw = sw[sw != no_switch_value]
        self.switch2D = switch2D
        try:
            self.switched_domain = switch2D >= self.sw[0]
        except:
            print("Error in the number of switches. It is likely you cropped the image too much")
            sys.exit()
        if is_remove_small_holes:
            self.switched_domain = remove_small_holes(~self.switched_domain, small_holes_area)
            self.switched_domain = ~self.switched_domain
        self.is_single_domain, self.initial_clusters, self.n_initial_clusters = self._is_single_domain(self.switched_domain, self.NNstructure)
        self.max_switch = self.max_switch_not_touching_edges()

    def _is_single_domain(self, domain, NNstructure):
        im, n_cluster = mahotas.label(domain, NNstructure)
       
        if n_cluster > 1:
            print("There are %d clusters" % n_cluster)
            is_single_domain = False
        else:
            # Check first if the domain is closed
            # using the anti-domain
            a_domain = remove_small_holes(~domain)
            im0, n_cluster0 = mahotas.label(a_domain, NNstructure)
            if n_cluster0 == 1:
                print("The domain is not closed")
                is_single_domain = False
            elif n_cluster0 >= 2:
                print("It is a single domain")
                is_single_domain = True
        return is_single_domain, im, n_cluster

    def _iterate_over_clusters(self, q, n_initial_clusters, show_plots=False):
        if n_initial_clusters == 1:
            # We need to split into two clusters to make it working
            row_c, col_c = [np.int(elem) for elem in nd.measurements.center_of_mass(q)]
            rows, cols = q.shape
            split_seq = [[0,row_c,0,cols], [row_c,rows,0,cols], [0,rows,0,col_c], [0,rows,col_c,cols]]
            for seq in split_seq:
                im = np.copy(q)
                r0,r1,c0,c1 = seq
                im[r0:r1,c0:c1] = False
                im, n = mahotas.label(im, self.NNstructure)
                if n > 1:
                    if show_plots:
                        fig,axs = plt.subplots(1,2,sharex=True, sharey=True)
                        axs[0].imshow(im)
                        axs[1].imshow(q)
                        plt.show()
                    break
        else:
            im, n_cluster0 = mahotas.label(q, self.NNstructure)
            n = n_initial_clusters
        # Search for the minimum distance
        for i, j in itertools.combinations(range(1, n+1), 2):
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
            #print(q_i, q_j)
            x_i, y_i = q_i.astype(int)
            x_j, y_j = q_j.astype(int)
            # Add a line between the two points
            #rr, cc, val = line_aa(x_i, y_i, x_j, y_j)
            rr, cc = line(x_i, y_i, x_j, y_j)
            q[rr, cc] = True
        if show_plots:
            fig,axs = plt.subplots(1,2,sharex=True, sharey=True)
            axs[0].imshow(im)
            axs[0].plot(cc,rr,'-w')
            axs[1].imshow(q)
            plt.show()
            print(x_i, y_i, x_j, y_j)
        return q

    def get_initial_domain(self, is_remove_small_holes=True, show_plots=False):
        """
        The detection of the initial domain has a lot of problems
        We need to differenciate between:
        1. a large domain with a few small clusters
        2. the final domain is spitted in part
            (i.e. parts of the wall do not move at all)
        
        Case 1:
            This is solved by removing small clusters
            with the remove_small_holes method.
            It seems to work quite generally
        Case 2:
            Apply the spatial.cKDTree which works perfectly
            for >3 initial clusters.
            For one initial domain, i.e. a domain with the form of a C
            we need to spit the domain into the four quadrants and find the min distance
            with spatial.cKDTree
            For two initial domains, we first find the min distance
            make the bridge, and consider it as having a single initial domain
        """
        print("%s Get the initial domain" % SPACE)
        q = np.copy(self.switched_domain)
        print("There are %i initial clusters" % self.n_initial_clusters)        
        if not self.is_single_domain:
            if self.n_initial_clusters == 2:
                # First calculate the bridge and make a single domain
                # then find the other bridge
                for i in [2,1]:
                    q = self._iterate_over_clusters(q, i)
            else:
                q = self._iterate_over_clusters(q, self.n_initial_clusters)
        # Removal of the border in this case requires 
        # the use of a hor/ver NNstructure, 
        # otherwise it does not work
        im, n_cluster = mahotas.label(~q, NNstructure)
        im = mahotas.labeled.remove_bordering(im)
        if n_cluster > 1:
            im, n_cluster = mahotas.labeled.relabel(im)
            size_clusters = mahotas.labeled.labeled_size(im)[1:]
            index = size_clusters.argmax()
            #size_central_domain = size_clusters[index]
            im = im == index + 1
        if show_plots:
            fig = plt.figure()
            ax = plt.gca()
            ax.imshow(im)
            plt.show()
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
    filename = "/data/src_backup/pyAvalanches/switch2D_05.pkl"
    with open(filename, 'rb') as f:
        switch2D = pickle.load(f)
    domain = Domains(switch2D)
    plt.figure()
    plt.imshow(domain.initial_clusters, interpolation='None')
    plt.figure()
    initial_domain = domain.get_initial_domain()
    plt.imshow(initial_domain, interpolation='None')
    plt.show()
