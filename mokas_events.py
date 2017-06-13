import sys, os
import pickle
import numpy as np
import mahotas
import getLogDistributions as gLD
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import measure
import scipy.ndimage as nd

def pixels_at_edges(cluster, with_diagonal=True):
    """
    Finds the pixels at the edges of a compact cluster
    """
    rolled = np.roll(cluster, 1, axis=0)          # shift down
    rolled[0, :] = False
    z = np.logical_or(cluster, rolled)
    rolled = np.roll(cluster, -1, axis=0)         # shift up 
    rolled[-1, :] = False
    z = np.logical_or(z, rolled)
    rolled = np.roll(cluster, 1, axis=1)          # shift right
    rolled[:, 0] = False
    z = np.logical_or(z, rolled)
    rolled = np.roll(cluster, -1, axis=1)         # shift left
    rolled[:, -1] = False
    z = np.logical_or(z, rolled)
    # ###########################
    if with_diagonal:
        for shift in [1,-1]:
            rolled0 = np.roll(cluster, shift, axis=0)
            index = shift - 1 * (shift==1)
            rolled0[index, :] = False
            for shift2 in [-1,1]:
                rolled = np.roll(rolled0, shift2, axis=1)
                index = shift2 - 1 * (shift2==1)
                rolled[:, index] = False
                z = np.logical_or(z, rolled) 

    # rolled = np.roll(cluster, 1, axis=0)          # shift down and left
    # rolled[0, :] = False
    # rolled = np.roll(rolled, -1, axis=1)
    # rolled[:, -1] = False
    # z = np.logical_or(z, rolled)
    # # ###########################
    # rolled = np.roll(cluster, 1, axis=0)          # shift down and right
    # rolled[0, :] = False
    # rolled = np.roll(rolled, 1, axis=1)
    # rolled[:, 0] = False
    # z = np.logical_or(z, rolled)
    # # ###########################
    # rolled = np.roll(cluster, -1, axis=0)          # shift up and left
    # rolled[-1, :] = False
    # rolled = np.roll(rolled, -1, axis=1)
    # rolled[:, -1] = False
    # z = np.logical_or(z, rolled)
    # # ###########################
    # rolled = np.roll(cluster, -1, axis=0)          # shift up and right
    # rolled[0, :] = False
    # rolled = np.roll(rolled, 1, axis=1)
    # rolled[:, 0] = False
    # z = np.logical_or(z, rolled)
    return z - cluster


class PlotEventsAndCluster:
    """
    class for plotting
    """
    def plot_size_distributions(self, events_sizes, cluster_sizes):
        fig, axs = plt.subplots(1, 1) # Distributions of events and clusters
        fig.suptitle(self.title, fontsize=30)
        # Calculate and plot the distributions of clusters and clusters
        for sizes, label in zip([events_sizes, cluster_sizes], ['events', 'clusters']):
            x, y, yerr = gLD.logDistribution(sizes, log_step=0.1, 
                                   first_point=1., normed=True)
            # Plots of the distributions
            axs.loglog(x, y,'o', label=label)
            axs.errorbar(x, y, yerr, fmt=None)
            if label == 'events':
                axs.loglog(x, 0.25 * x**-1.17 * np.exp(-x/100),'-', label=r'S^{-1.17} exp(-S/50)')
            elif label == 'clusters':
                axs.loglog(x, 0.25 * x**-1.17,'-', label=r'S^{-1.17} exp(-S/50)')
        axs.legend()
        axs.grid(True)

    def average_size_vs_duration_of_clusters(self, durations, sizes):
        fig, ax = plt.subplots(1, 1)
        if durations[0] == 0:
            durations = durations[1:]
            sizes = sizes[1:]

        ax.loglog(durations, sizes,'o')
        ax.set_xlabel("Duration (frames)")
        ax.set_ylabel("Sizes (pixels area)")
        ax.grid(True)
        log_sizes = np.log10(sizes)
        log_durations = np.log10(durations)
        res = np.polyfit(log_durations, log_sizes, 1)
        gamma = res[0]
        c0 = 10**res[1]
        ax.loglog(durations, c0*durations**gamma, label=str(gamma))
        plt.legend()
        plt.show()
    

class EventsAndClusters():
    def __init__(self, switch2D, set_init_time=True, NNstructure=None,
                post_processing=None, row_data2D=None):
        """
        switch2D is the map of switches
        or the filename
        Parameters:
        ===========
        switch2D : 2D map of switches
        set_init_time : True|False 
            Define the cluster using the initial|final value of the event
        """
        if isinstance(switch2D, np.ndarray):
            print("Data collected")
            self.switch2D = switch2D
        elif isinstance(switch2D, str):
            print("File %s loading" % switch2D)
            with open(switch2D, 'rb') as f:
                self.switch2D = pickle.load(f)
            print("Done")
        self.set_init_time = set_init_time
        if NNstructure is None:
            NN = 3
            self.NNstructure = np.ones((NN,NN))
        else:
            self.NNstructure = NNstructure
        if post_processing and row_data2D is not None:
            self.switch2D = self._post_processing(switch2D, row_data2D)

        self.switches = np.unique(self.switch2D)
        if self.switches[0] < 0:
            self.fillValue = self.switches[0]
            self.switches = self.switches[1:]

        self.is_events_and_clusters = False


    def get_events_and_clusters(self, min_cluster_size=None, cluster_limits=None):
        print("Calculation of events")
        self.events_sizes = self._get_events()
        print(len(self.events_sizes))
        ##########################
        print("Calculation of clusters")
        self.cluster2D = self._get_cluster2D('limits', min_cluster_size, cluster_limits)
        self.cluster_switches = np.unique(self.cluster2D)[1:]
        size_and_durations = self._get_clusters_size_and_duration(self.cluster2D)
        self.cluster_sizes, self.cluster_durations = size_and_durations
        print("Calculation of average clusters")
        averages = self._get_average_clusters(*size_and_durations)
        self.average_cluster_sizes, self.average_cluster_durations = averages
        print("We have collected %i events and %i clusters" % (len(self.events_sizes), len(self.cluster_switches)))
        self.is_events_and_clusters = True

    def _get_events(self):
        """
        get the statistics of the events
        """
        events_sizes = np.array([])
        for switch in self.switches:
            q = self.switch2D == switch
            im, n_cluster = mahotas.label(q, self.NNstructure)
            sizes = mahotas.labeled.labeled_size(im)[1:]
            events_sizes = np.concatenate((events_sizes, sizes))
        return events_sizes

    def _get_cluster2D(self, method='limits', min_cluster_size=None, cluster_limits=None):
        """
        method : str [limits|edges]
            'limits' considers the clusters within two values passes by cluster_limits
            'edges' considers all the clusters having adjacent pixels
        """
        if method == 'limits':
            return self._get_cluster2D_limits(min_cluster_size, cluster_limits)
        elif method == 'edges':
            return self._get_cluster2D_edges(min_cluster_size)
        else:
            print("Please, pass a criterium to get the clusters (egdes|limits)")

    def _get_cluster2D_limits(self, min_cluster_size=0, cluster_limits=None):
        """
        method to get the clusters define by the limits, i.e. the values of the
        switches which define the initial and final frame of the cluster
        This is a two-pass calculation:
        I : the clusters are calculated using the limits (cluster_limits) calculated elsewhere
        using a threshold
        II : add the switches between the end and the start of a cluster which are below the threshold
        """
        print("Using the limits")
        cluster2D = np.copy(self.switch2D)
        if self.set_init_time:
            istep = -1
        else:
            istep = 1
        # pass I
        for sw_in, sw_fin in cluster_limits:
            print(sw_in, sw_fin)
            if self.set_init_time:
                sw_in, sw_fin = sw_fin, sw_in
            switches = range(sw_in, sw_fin, istep)
            # loop for the switches within the limits
            for sw0 in switches:
                q = cluster2D == sw0
                sw_next = sw0 + istep
                sub_clusters, n_sub_clusters = mahotas.label(q, self.NNstructure)
                for i in range(1, n_sub_clusters+1):
                    cluster = sub_clusters == i
                    # Check on the size of EACH cluster seems not needed
                    #size_cluster = np.sum(cluster)
                    #if size_cluster < min_cluster_size:
                    #    break
                    cluster_edge = pixels_at_edges(cluster)
                    switches_at_edge = np.extract(cluster_edge, cluster2D)
                    if sw_next in switches_at_edge:
                        cluster2D[cluster] = sw_next
        # pass II
        cluster_switches = np.unique(cluster2D)[1:]
        n_cluster_limits = len(cluster_limits)
        for i, (sw_in, sw_fin) in enumerate(cluster_limits):
            if self.set_init_time:
                main_cluster_size = np.sum(cluster2D == sw_in)
            else: 
                main_cluster_size = np.sum(cluster2D == sw_fin)
            if i == 0:
                sw0 = cluster_switches[0] - 1
            if i == n_cluster_limits - 1:
                sw1 = cluster_switches[-1]
            else:
                sw1 = cluster_limits[i+1][0]
            q = np.logical_and((cluster2D > sw0), (cluster2D < sw1))
            sw0 = sw_fin
            
            sub_clusters, n_sub_clusters = mahotas.label(q, self.NNstructure)
            for i in range(1, n_sub_clusters+1):
                cluster = sub_clusters == i
                size_cluster = np.sum(cluster)
                if size_cluster > main_cluster_size:
                    if self.set_init_time:
                        cluster2D[cluster] = sw_in
                    else:
                        cluster2D[cluster] = sw_fin
                main_cluster_size = size_cluster # update the largest cluster size

        return cluster2D


    def _get_cluster2D_edges(self, min_cluster_size):
        """
        statistics of clusters
        """
        print(min_cluster_size)

        cluster2D = np.copy(self.switch2D)
        if self.set_init_time:
            istep = -1
        else:
            istep = 1
        for sw0 in self.switches[::istep]:
            q = cluster2D == sw0
            sw_next = sw0 + istep
            clusters, n_cluster = mahotas.label(q, self.NNstructure)
            for i in range(1, n_cluster+1):
                cluster = clusters == i
                size_cluster = np.sum(cluster)
                if size_cluster < min_cluster_size:
                    continue
                cluster_edge = pixels_at_edges(cluster)
                # Get the values of the switches around the cluster
                switches_at_edge = np.extract(cluster_edge, cluster2D)
                # Check if any of the neighbours is sw0 + istep
                # and set the original cluster to sw0 + istep
                if sw_next in switches_at_edge:
                    cluster2D[cluster] = sw_next
        return cluster2D

    def _get_clusters_size_and_duration(self, clusters):
        # Calculus of the cluster sizes and durations
        print("Calculation cluster sizes and durations")
        cluster_sizes = np.array([])
        cluster_durations = np.array([])
        cluster_switches = np.unique(clusters)[1:]
        for switch in cluster_switches:
            clusters, n_cluster = mahotas.label(clusters == switch, self.NNstructure)
            sizes = mahotas.labeled.labeled_size(clusters)[1:]
            durations = np.array([np.max(np.extract(clusters == i, self.switch2D)) - switch for i in range(1, n_cluster+1)])
            cluster_sizes = np.concatenate((cluster_sizes, sizes))
            cluster_durations = np.concatenate((cluster_durations, durations))
        assert len(cluster_sizes) == len(cluster_durations)
        return cluster_sizes, cluster_durations

    def _get_average_clusters(self, cluster_sizes, cluster_durations):
        # Calculus of the cluster sizes and durations
        average_cluster_durations = np.unique(cluster_durations)
        average_cluster_sizes = []
        for duration in average_cluster_durations:
            size = np.mean(np.extract(cluster_durations == duration, cluster_sizes))
            average_cluster_sizes.append(size)
        average_cluster_sizes = np.array(average_cluster_sizes)
        return average_cluster_sizes, average_cluster_durations


    def plot_maps(self, cmap='pastel', zoom_in_data=True, 
                    fig=None, axs=None, title=None,
                    with_cluster_number=False):

        if not self.is_events_and_clusters:
            self.get_events_and_clusters()

        if cmap == 'pastel' or cmap == 'random':
            n_colors = self.switches[-1] - self.switches[0] + 1
            clrs = np.random.rand(n_colors, 3) 
            if cmap == 'pastel':
                clrs = (clrs + [1,1,1])/2
            clrs[0] = [0,0,0]
            self.cmap = mpl.colors.ListedColormap(clrs)
        else:
            self.cmap = cmap

        if zoom_in_data:
            rows_mean_sw = np.mean(self.switch2D, axis=1)
            jj = np.where(rows_mean_sw != self.fillValue)
            i0, i1 = np.min(jj) - 20, np.max(jj) + 20
            rows, cols = self.switch2D.shape
            if i0 < 0:
                i0 = 0
            if i1 > rows:
                i1 = rows
            switch2D = self.switch2D[i0:i1+1,:]
            cluster2D = self.cluster2D[i0:i1+1,:]
        else:
            switch2D = self.switch2D
            cluster2D = self.cluster2D
        cluster_switches = np.unique(self.cluster2D)[1:]
        
        # Plot
        if not fig:
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True) # ColorImages of events and sizes
            ax0, ax1 = axs[0], axs[1]
        else:
            ax0, ax1 = axs
        ax0.imshow(switch2D, cmap=self.cmap)
        ax1.imshow(cluster2D, cmap=self.cmap)
        rows, cols = cluster2D.shape
        ax0.axis((0,cols,rows,0))
        font = {'weight': 'normal', 'size': 8}
        for i in cluster_switches:
            cluster = cluster2D == i
            cnts = measure.find_contours(cluster, 0.5)
            for cnt in cnts:
                X,Y = cnt[:,1], cnt[:,0]
                for ax in [ax0, ax1]:
                    ax.plot(X, Y, c='k', antialiased=True, lw=1)
            if with_cluster_number:
                # Calculate the distance map and find the indexes of the max
                d = mahotas.distance(cluster)
                yc, xc = np.unravel_index(d.argmax(), d.shape)
                # print("cluster %i: (%i, %i)" % (i, xc, yc))
                ax1.text(xc, yc, str(i), horizontalalignment='center',
                    verticalalignment='center', fontdict=font)
        if title:
            fig.suptitle(title, fontsize=30)


#############################################################################

if __name__ == "__main__":
    try:
        mtype = sys.argv[1]
    except:
        mtype = 'Irr16'
    NNstructure = np.ones((3,3))
    qq = np.array([[13, 13, 13,  8,  8,  3],
       [12, 14,  8,  6,  3,  3],
       [ 9,  4,  4,  4,  3, -1],
       [15, 15, 15,  2, -1, -1],
       [15, 15, -1, -1, -1, -1],
       [-1, -1, -1, -1, -1, -1]], dtype=np.int32)
    
    qq2 = np.array([[13, 13, 13,  8,  7,  3],
       [12, 14,  8,  6,  3,  3],
       [ 8,  4,  4,  4,  3, -1],
       [15,  9, 15,  2, -1, -1],
       [15, 15,  2, -1, -1, -1],
       [-1, 2, -1, -1, -1, -1]], dtype=np.int32)
    filenamepkl = "switchTimes2D.pkl"
    #filename = "switch2D_test.pkl"
    if mtype == "NonIrr":
        rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr"
        subDir = "NonIrr_0.095A_3s"
        experiments = (1,2,4)
    elif mtype == "Irr16":
        rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC_16e8He+/"
        subDir = "Irr_16e8He+_0.116A_3s"
        experiments = (2,3,4,5,6,7,8,9,10)
    events_sizes = np.array([])
    cluster_sizes = np.array([])
    cluster_durations = np.array([])
    
    for i in experiments:
        sub_Dir = "%s_%s" % (str(i).rjust(2,"0"), subDir)
        filename = os.path.join(rootDir, sub_Dir, filenamepkl)
        events = Events(filename, NNstructure=NNstructure)
        events.get_events_and_clusters()
        events.plot_maps(title=sub_Dir)
        events_sizes = np.concatenate((events_sizes, events.events_sizes))
        cluster_sizes = np.concatenate((cluster_sizes, events.cluster_sizes))
        cluster_durations = np.concatenate((cluster_durations, events.cluster_durations))

    average_cluster_sizes, average_cluster_durations = events._get_average_clusters(cluster_sizes, cluster_durations)
    title = "%s%s" % (str(experiments), subDir)
    plotEvents = PlotEvents(title=title)
    plotEvents.size_distributions(events_sizes, events.cluster_sizes)
    plotEvents.average_size_vs_duration_of_clusters(average_cluster_durations, average_cluster_sizes)