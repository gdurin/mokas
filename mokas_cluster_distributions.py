import sys, os
import numpy as np
import getLogDistributions as gLD
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from collections import OrderedDict, defaultdict
import h5py
import getAxyLabels as gal
import pandas as pd
from skimage import measure
import heapq
import mokas_cluster_methods as cmet
import mahotas
from mokas_colors import get_liza_colors


class Clusters:

    def __init__(self, hdf5_filename, group, n_experiments, min_size=5,
                skip_first_clusters=0, motion='downward'):
        """
        it is important to set a min_size of the cluster. 
        The value of 10 seems reasonable, but can larger
        """
        self.min_size = min_size
        self.skip_first_clusters = skip_first_clusters
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
        if motion == 'downward':
            self.direction = 'Bottom_to_top'


    def get_experiment_stats(self):
        print("Get the statistics for each experiment")
        self.cluster_data = dict()
        self.sub_cluster_data = dict()
        self.cluster2D_color = dict()
        ######################################
        for n_exp in self.cluster2D:
            print("Experiment: %i" % n_exp)
            cluster_data = defaultdict(list)
            sub_cluster_data = defaultdict(list)
            ############################################
            cluster2D = self.cluster2D[n_exp]
            self.cluster2D_color[n_exp] = np.copy(cluster2D)
            cluster_switches = np.unique(cluster2D)[self.skip_first_clusters+1:]
            cluster = np.logical_and((cluster2D > 0), cluster2D < cluster_switches[0])
            rows, cols = cluster2D.shape
            for switch in cluster_switches:
                #print(switch)
                cluster0 = cluster2D == switch
                cluster += cluster0
                cluster_type = gal.getAxyLabels(cluster0, self.direction, 1)[0]
                # We need to check if the cluster is touching the top/bottom edge
                # i.e. we check the last two elements of the cluster_type
                if '1' in cluster_type[2:]:
                    self.cluster2D_color[n_exp][cluster0] = -1
                    continue
                angle_left, angle_right = self.get_angles(cluster)
                # Get and plor the curvature
                cnts = self._select_contour(cluster)
                X, Y = cnts[:,1], cnts[:,0]
                a, b, c = np.polyfit(X, Y, 2)
                r_curvature = -1./a
                x_v = - b / (2 * a)
                y_v = - b * b / (4 * a) + c
                c_right = a * cols * cols + b * cols + c
                row_of_min = np.max(Y) # Yeah, it is correct
                y_left, y_right = Y[0], Y[-1]
                if X[0] != 0:
                    y_left, y_right = y_right, y_left
                delta_left = np.abs((y_left) - row_of_min)
                delta_right = np.abs((y_right) - row_of_min)
                l_front = cmet.get_length(cnts)

                # Here there is a problem
                # Cluster0 can be made of 2 or more subclaster (this is not uncommon)
                # So we need to loop over the sub_clusters
                sub_clusters, n_sub_clusters = mahotas.label(cluster0, np.ones((3,3)))
                # Save the data
                cluster_data['n_exp'].append(n_exp)
                cluster_data['switch'].append(switch)
                cluster_data['type'].append(cluster_type)
                cluster_data['area'].append(np.sum(cluster0))
                cluster_data['n_sub_cl'].append(n_sub_clusters)
                cluster_data['a_01'].append(angle_left)
                cluster_data['a_10'].append(angle_right)
                cluster_data['r_curv'].append(r_curvature)
                cluster_data['x_v'].append(x_v)
                cluster_data['y_v'].append(y_v)
                cluster_data['c_left'].append(c)
                cluster_data['c_right'].append(c_right)
                cluster_data['delta_left'].append(delta_left)
                cluster_data['delta_right'].append(delta_right)
                cluster_data['l_front'].append(l_front)
                # Loop over the sub_clusters
                for label in range(1, n_sub_clusters+1):
                    sub_cluster = sub_clusters == label
                    #print("Sub_cluster n. %i" % label)
                    # Check again the cluster_type
                    sub_cluster_type = gal.getAxyLabels(sub_cluster, self.direction, 1)[0]
                    # Update the color map
                    q = self.cluster2D_color[n_exp]
                    q[sub_cluster] = np.int(sub_cluster_type[:2], 2)
                    self.cluster2D_color[n_exp] = q
                    area = np.sum(sub_cluster)
                    if area < self.min_size:
                        continue
                    #print(sub_cluster_type)
                    #print(n_exp, switch, cluster_type)
                    # Calculate the lengths of the subcluster
                    # L is the linear distance between the 
                    l0, l1, L_linear, success = cmet.get_upper_and_lower_contour(sub_cluster, sub_cluster_type,
                                                self.ref_point, motion=self.motion)
                    if success:
                        l0, l1 = cmet.get_length(l0), cmet.get_length(l1)
                    sub_cluster_data['switch'].append(switch)
                    sub_cluster_data['type'].append(sub_cluster_type)
                    sub_cluster_data['area'].append(area)
                    sub_cluster_data['l0'].append(l0)
                    sub_cluster_data['l1'].append(l1)
                    sub_cluster_data['L_linear'].append(L_linear)

            cluster_cols = ['n_exp', 'switch', 'type', 'area', 'n_sub_cl',
                            'a_10', 'a_01', 'r_curv', 'x_v', 'y_v',
                            'c_left', 'c_right', 'delta_left', 'delta_right',
                            'l_front']
            df = pd.DataFrame.from_dict(cluster_data)
            self.cluster_data[n_exp] = df[cluster_cols]
            sub_cluster_cols = ['switch', 'type', 'area', 'l0', 'l1', 'L_linear']
            df = pd.DataFrame.from_dict(sub_cluster_data)
            self.sub_cluster_data[n_exp] = df[sub_cluster_cols]
            del df
            
    def plot_global_stats(self, color='red', whiter=.5):
        colors, cmap = get_liza_colors(color, whiter)
        # Do it only after exp and global stats
        fig, axs = plt.subplots(2,2)
        q = self.all_clusters
        # Plot the hist of the curvature
        ax = axs[0,0]
        c = q.r_curv
        n, bins, patches = ax.hist(c, bins=100, facecolor='green', alpha=0.75)
        ax.set_xlabel("Radius of Curvature")
        ax.set_title(r'${R.\ of\ Curvature:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        # Plot the pie
        ax = axs[0,1]
        labels = [Axy[:2] for Axy in self.Axy_types]
        fracs = [q[q.type==Axy].area.sum() for Axy in self.Axy_types]
        explode=(0.05, 0, 0, 0)
        ax.pie(fracs, explode=explode, labels=labels, colors=colors[1:],
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        # Plot the hist of the left angle
        ax = axs[1,0]
        c = q.a_10
        n, bins, patches = ax.hist(c, bins=100, facecolor='blue', alpha=0.75)
        ax.set_xlabel("Left angle (degree)")
        ax.set_title(r'$\mathrm{Left\ angle:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        ax = axs[1,1]
        c = q.a_01
        n, bins, patches = ax.hist(c, bins=100, facecolor='purple', alpha=0.75)
        ax.set_xlabel("Right angle (degree)")
        ax.set_title(r'$\mathrm{Right\ angle:}\ \mu=%.2f,\ \sigma=%.2f$' % (c.mean(), c.std()))
        plt.suptitle(self.title, fontsize=20)
        ##########
        plt.show()

    def plot_cluster_maps(self, color='red', whiter=0.5, zoom_in_data=True):
        Ncols = 5
        colors, cmap = get_liza_colors(color, whiter)
        n_experiments = len(self.cluster2D)
        n_figs = np.int(np.ceil(float(n_experiments)/Ncols))
        figs, axs_i = [], []
        if n_experiments >= Ncols:
            n = Ncols
        else:
            n = n_experiments
        for i in range(n_figs):
            fig, axs = plt.subplots(1, n, sharey=False, squeeze=False)
            figs.append(fig)
            axs_i.append(axs)

        dh_max = 0
        h_min = []
        for i, n_exp in enumerate(self.cluster2D_color):
            cluster2D_color = self.cluster2D_color[n_exp]
            cluster2D = self.cluster2D[n_exp]
            n_fig = np.int(i/Ncols)
            n_ax = i % Ncols
            ax = axs_i[n_fig][0,n_ax]
            if zoom_in_data:
                rows_mean_sw = np.mean(cluster2D, axis=1)
                jj = np.where(rows_mean_sw != -1)
                i0, i1 = np.min(jj) - 20, np.max(jj) + 20
                rows, cols = cluster2D.shape
                if i0 < 0:
                    i0 = 0
                if i1 > rows:
                    i1 = rows
                dh = i1 - i0
                if dh > dh_max:
                    dh_max = dh
                h_min.append(i0)
            ax.imshow(cluster2D_color, cmap=cmap, interpolation='nearest')
            for switch in np.unique(cluster2D)[1:]:
                cluster = cluster2D==switch
                cnts = measure.find_contours(cluster, 0.5)
                for cnt in cnts:
                    X,Y = cnt[:,1], cnt[:,0]
                    ax.plot(X, Y, c='k', antialiased=True, lw=1)
            ax.set_title("Exp. n. %i" % n_exp)
        if zoom_in_data:
            for i in range(n_experiments):
                n_fig = np.floor(i/np.float(Ncols)).astype(np.int)
                ax = axs_i[n_fig][0,i%Ncols]
                print(n_fig, i%Ncols)
                ax_coords = 0, cols, h_min[i] + dh_max, h_min[i]
                ax.axis(ax_coords)
        plt.show()


    def get_global_stats(self):
        print("Calculating the global stats")
        self.all_clusters = pd.concat([self.cluster_data[cl] for cl in self.cluster_data])
        self.all_sub_clusters = pd.concat([self.sub_cluster_data[cl] for cl in self.sub_cluster_data])
        self.total_area = np.sum(self.all_clusters.area)
        for Axy in self.Axy_types:
            q = self.all_sub_clusters[self.all_sub_clusters.type==Axy]['area']
            s = q.sum()
            print("Type %s: fraction of area covered is %.2f" % 
                (Axy, s/float(self.total_area)*100.))
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
            i1 = len(lens) - 1 - lens[::-1].index(l_values[0])
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
        # Example: 
        set_current, n_wire = sys.argv[2:]
        fname = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_old/Ta_CoFeB_MgO_wires_IEF_old.hdf5"
        grp0  = "20um/%s/10fps/wire%s" % (set_current, n_wire)
        if not os.path.isfile(fname):
            print("Chech the path")
            sys.exit()
        n_experiments = range(1,11)
        color = 'red'
        clusters = Clusters(fname, grp0, n_experiments, skip_first_clusters=0, min_size=3)
        clusters.get_experiment_stats()
        clusters.get_global_stats()
        clusters.plot_global_stats(color=color)
        clusters.plot_cluster_maps(color=color, whiter=0.5)