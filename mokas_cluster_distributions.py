import sys, os
import numpy as np
import getLogDistributions as gLD
import matplotlib.pyplot as plt
from collections import OrderedDict
import h5py
import getAxyLabels as gal
import pandas as pd
from skimage import measure
import heapq
import mokas_cluster_methods as cmet


class Clusters:

    def __init__(self, hdf5_filename, group, n_experiments, 
                motion='downward'):
        self._fname = hdf5_filename
        p, filename = os.path.split(hdf5_filename)
        filename, ext = os.path.splitext(filename)
        self.title = filename + " - " + " - ".join(group.split("/")) + " - %i exps." % len(n_experiments)
        self._baseGroup = group
        self.cluster2D = OrderedDict()
        with h5py.File(self._fname, 'a') as f:
            for n_exp in n_experiments:
                cluster_dataset = self._baseGroup + "/%i/cluster2D" % n_exp
                grp0 = f[cluster_dataset]
                self.cluster2D[n_exp] = grp0[...]
        rows, cols = self.cluster2D[n_exp].shape
        # All the figure have to be roated to get the motion downward
        self.ref_point = (0, cols//2)
        self.motion = motion
        self.Axy_types = ['0000', '0100', '1000','1100']
        self.colors = {'0000': 'r', '1000': 'b', '0100': 'c', '1100': 'g'}
        self.cluster_data = OrderedDict()
        self.get_experiment_stats(skip_first_clusters=1)
        self.get_global_stats()
        self.plot_global_stats(10)

    def get_experiment_stats(self, direction='Bottom_to_top', skip_first_clusters=1):
        print("Get the statistics for each experiment")
        ######################################
        for n_exp in self.cluster2D:
            print("Experiment: %i" % n_exp)
            cluster_types = []
            cluster_sizes = []
            angles_left = []
            angles_right = []
            curvature = []
            largest_sizes = []
            l_up = []
            l_down = []
            self.cluster_data[n_exp] = pd.DataFrame()
            cluster2D = self.cluster2D[n_exp]
            switches = np.unique(cluster2D)[skip_first_clusters:]
            cluster = np.zeros_like(cluster2D).astype(bool)
            for switch in switches:
                cluster0 = cluster2D == switch
                cluster_type = gal.getAxyLabels(cluster0, direction, 1)[0]
                # We need to check if the cluster is touching the top/bottom edge
                if '1' in cluster_type[2:]:
                    switches = switches[switches<switch]
                    break
                cluster_types.append(cluster_type)
                size = np.sum(cluster0)
                cluster_sizes.append(size)
                #print(n_exp, switch, cluster_type)
                cluster += cluster0
                angle_left, angle_right = self.get_angles(cluster)
                angles_right.append(angle_right)
                angles_left.append(angle_left)
                # Get and plor the curvature
                cnts = self._select_contour(cluster)
                X, Y = cnts[:,1], cnts[:,0]
                z = np.polyfit(X, Y, 2)
                curvature.append(-1./z[0])
                # Calculate the lengths
                im, largest_size = cmet.largest_cluster(cluster0, NNstructure=np.ones((3,3)))
                largest_sizes.append(largest_size)
                l0, l1, im_corners = cmet.get_upper_and_lower_contour(cluster0, self.motion, self.ref_point)
                l0, cv = cmet.get_lenght_and_curvature(l0)
                l1, cv = cmet.get_lenght_and_curvature(l1)
                l_up.append(l0)
                l_down.append(l1)
            self.cluster_data[n_exp]['switches'] = switches
            self.cluster_data[n_exp]['types'] = cluster_types
            self.cluster_data[n_exp]['sizes'] = cluster_sizes
            self.cluster_data[n_exp]['angles_left'] = angles_left
            self.cluster_data[n_exp]['angles_right'] = angles_right
            self.cluster_data[n_exp]['curvature'] = curvature
            self.cluster_data[n_exp]['largest_sizes'] = largest_sizes
            self.cluster_data[n_exp]['l_up'] = l_up
            self.cluster_data[n_exp]['l_down'] = l_down
            
    def plot_global_stats(self, min_size=0):
        # Do it only after exp and global stats
        fig, axs = plt.subplots(2,2)
        q = self.all_clusters[self.all_clusters.sizes>min_size]
        # Plot the hist of the curvature
        ax = axs[0,0]
        c = q.curvature
        n, bins, patches = ax.hist(c, bins=100, facecolor='green', alpha=0.75)
        ax.set_xlabel("Radio of Curvature")
        ax.set_title(r'$\mathrm{R Curvature:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        # Plot the pie
        ax = axs[0,1]
        labels = [Axy[:2] for Axy in self.Axy_types]
        fracs = [q[q.types==Axy].sizes.sum() for Axy in self.Axy_types]
        explode=(0.05, 0, 0, 0)
        ax.pie(fracs, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        # Plot the hist of the left angle
        ax = axs[1,0]
        c = q.angles_left
        n, bins, patches = ax.hist(c, bins=100, facecolor='blue', alpha=0.75)
        ax.set_xlabel("Left angle (degree)")
        ax.set_title(r'$\mathrm{Left angle:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        ax = axs[1,1]
        c = q.angles_right
        n, bins, patches = ax.hist(c, bins=100, facecolor='purple', alpha=0.75)
        ax.set_xlabel("Right angle (degree)")
        ax.set_title(r'$\mathrm{Right angle:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        plt.suptitle(self.title, fontsize=20)
        plt.show()

    def get_global_stats(self):
        print("Calculating the global stats")
        self.all_clusters = pd.concat([self.cluster_data[cl] for cl in self.cluster_data])
        self.total_size = np.sum(self.all_clusters.sizes)
        for Axy in self.Axy_types:
            q = self.all_clusters[self.all_clusters.types==Axy]['sizes']
            s = q.sum()
            print("Type %s: fraction of area covered is %.2f" % 
                (Axy, s/float(self.total_size)*100.))
        self.is_global_stats = True
        # TODO: plot hist of angles
        # distributions of Pxy and Axy


    def get_angles(self, cluster, n_fit=30):
        cnts = self._select_contour(cluster)
        X, Y = cnts[:,1], cnts[:,0]
        ####################################
        # Left edge
        ####################################
        z = np.polyfit(X[:n_fit], Y[:n_fit], 1)
        z0 = np.arctan(z[0])
        #print("left: %f" % z0)
        angle_left = np.abs(z0) * 180 / np.pi
        #mksize = 15./700*(np.sum(cluster0)) + 1
        #plt.plot(switch, angle, 'o', c=colors[cluster_type], markersize=mksize, alpha=0.5)
        ####################################
        # Right edge
        ####################################
        z = np.polyfit(X[-n_fit:], Y[-n_fit:], 1)
        z0 = np.arctan(z[0])
        #print("right: %f" % z0)
        angle_right = abs(z0) * 180 / np.pi
        return angle_left, angle_right
        

    def _select_contour(self, cluster, position='bottom'):
        """
        select the two longest contours in a list
        then get the one corresponding to the position
        """
        cnts = measure.find_contours(cluster, 0.5)
        lens = [len(cnt) for cnt in cnts]
        l_values = heapq.nlargest(2, lens)
        i0, i1 = [lens.index(l) for l in l_values]
        if l_values[0] == l_values[1]:
            i1 = len(lengths) - 1 - lengths[::-1].index(l_values[0])
        Y0, Y1 = cnts[i0][:,0], cnts[i1][:,0]
        m0, m1 = np.mean(Y0), np.mean(Y1)
        if m0 > m1:
            l_bottom = cnts[i0]
            l_top = cnts[i1]
        else:
            l_bottom = cnts[i1]
            l_top = cnts[i0]
        if position == 'bottom':
            return l_bottom
        else:
            return l_top    

if __name__ == "__main__":
    plt.close("all")
    imParameters = {}
    choice = sys.argv[1]
        # As on May 24, this is the example to followcl
    # Added hdf5=True
    if choice == 'IEF_old_20um':
        set_current, n_wire = sys.argv[2:]
        fname = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_old/Ta_CoFeB_MgO_wires_IEF_old.hdf5"
        grp0  = "20um/%s/10fps/wire%s" % (set_current, n_wire)
        if not os.path.isfile(fname):
            print("Chech the path")
            sys.exit()
        n_experiments = range(1,2)
        clusters = Clusters(fname, grp0, n_experiments)
