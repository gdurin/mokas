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
    return np.logical_xor(z, cluster)



# class PlotEventsAndCluster:
#     """
#     class for plotting
#     """
#     def plot_size_distributions(self, events_sizes, cluster_sizes):
#         fig, axs = plt.subplots(1, 1) # Distributions of events and clusters
#         fig.suptitle(self.title, fontsize=30)
#         # Calculate and plot the distributions of clusters and clusters
#         for sizes, label in zip([events_sizes, cluster_sizes], ['events', 'clusters']):
#             x, y, yerr = gLD.logDistribution(sizes, log_step=0.1, 
#                                    first_point=1., normed=True)
#             # Plots of the distributions
#             axs.loglog(x, y,'o', label=label)
#             axs.errorbar(x, y, yerr, fmt=None)
#             if label == 'events':
#                 axs.loglog(x, 0.25 * x**-1.17 * np.exp(-x/100),'-', label=r'S^{-1.17} exp(-S/50)')
#             elif label == 'clusters':
#                 axs.loglog(x, 0.25 * x**-1.17,'-', label=r'S^{-1.17} exp(-S/50)')
#         axs.legend()
#         axs.grid(True)

#     def average_size_vs_duration_of_clusters(self, durations, sizes):
#         fig, ax = plt.subplots(1, 1)
#         if durations[0] == 0:
#             durations = durations[1:]
#             sizes = sizes[1:]

#         ax.loglog(durations, sizes,'o')
#         ax.set_xlabel("Duration (frames)")
#         ax.set_ylabel("Sizes (pixels area)")
#         ax.grid(True)
#         log_sizes = np.log10(sizes)
#         log_durations = np.log10(durations)
#         res = np.polyfit(log_durations, log_sizes, 1)
#         gamma = res[0]
#         c0 = 10**res[1]
#         ax.loglog(durations, c0*durations**gamma, label=str(gamma))
#         plt.legend()
#         plt.show()
    

class EventsAndClusters():
    def __init__(self, switch2D, NNstructure=None,
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


    # def get_events_and_clusters(self, min_cluster_size=None, cluster_limits=None):
    #     print("Calculation of events")
    #     self.events_sizes = self._get_events()
    #     print(len(self.events_sizes))
    #     ##########################
    #     print("Calculation of clusters")
    #     self.cluster2D = self.get_cluster2D('limits', min_cluster_size, cluster_limits)
    #     self.cluster_switches = np.unique(self.cluster2D)[1:]
    #     size_and_durations = self._get_clusters_size_and_duration(self.cluster2D)
    #     self.cluster_sizes, self.cluster_durations = size_and_durations
    #     print("Calculation of average clusters")
    #     averages = self._get_average_clusters(*size_and_durations)
    #     self.average_cluster_sizes, self.average_cluster_durations = averages
    #     print("We have collected %i events and %i clusters" % (len(self.events_sizes), len(self.cluster_switches)))
    #     self.is_events_and_clusters = True

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

    def get_cluster2D(self, method='limits', min_cluster_size=None, cluster_limits=None):
        """
        method : str [limits|edges]
            'limits' considers the clusters within two values passes by cluster_limits
            'edges' considers all the clusters having adjacent pixels
        """
        self.is_events_and_clusters = True
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
        using a treshold
        II : add the switches between the end and the start of a cluster which are below the threshold
        """
        print("Using the limits")
        #cluster2D_end gives the end time of the cluster
        #cluster2D_start gives the start time of the cluster
        cluster2D_start = np.copy(self.switch2D)
        cluster2D_end = np.copy(self.switch2D)
        cluster_switches = np.unique(cluster2D_end)[1:]
        
        q = list(cluster_limits)
        _cluster_limits = np.array(q)
        
        # pass I
        for sw_in, sw_fin in _cluster_limits:
            print(sw_in, sw_fin)
            switches = range(sw_in, sw_fin, 1)
            # loop for the switches within the limits
            for sw0 in switches:
                q = cluster2D_end == sw0
                sw_next = sw0 + 1
                sub_clusters, n_sub_clusters = mahotas.label(q, self.NNstructure)
                for i in range(1, n_sub_clusters+1):
                    cluster = sub_clusters == i
                    # Check on the size of EACH cluster seems not needed
                    #size_cluster = np.sum(cluster)
                    #if size_cluster < min_cluster_size:
                    #    break
                    cluster_edge = pixels_at_edges(cluster)
                    switches_at_edge = np.extract(cluster_edge, cluster2D_end)
                    if sw_next in switches_at_edge:
                        cluster2D_end[cluster] = sw_next
                        cluster2D_start[cluster] = sw_in

        # Update the cluster_start
        # This is done in the line above!

        # for i, switch in enumerate(sw_fin):
        #     q = cluster2D_end == switch
        #     clusters, n_cluster = mahotas.label(q, self.NNstructure)
        #     for i in range(1, n_cluster+1):
        #         cluster = clusters == i
        #         cluster2D_start[cluster] = np.min(self.switch2D[cluster]) 

        # pass II
        # With the procedure above all the cluster2D_start have a value sw_in 
        
        n_cluster_limits = _cluster_limits.shape[0]
        for i, (sw_in, sw_fin) in enumerate(_cluster_limits):
            main_cluster_size = np.sum(cluster2D_start == sw_in)
            if i == 0:
                sw0 = cluster_switches[0] - 1
            if i == n_cluster_limits - 1:
                sw1 = cluster_switches[-1]
            else:
                sw1 = _cluster_limits[i+1][0]
            q = np.logical_and((cluster2D_start > sw0), (cluster2D_start < sw1))
            sw0 = sw_fin
            
            sub_clusters, n_sub_clusters = mahotas.label(q, self.NNstructure)
            for i in range(1, n_sub_clusters+1):
                cluster = sub_clusters == i
                size_cluster = np.sum(cluster)
                if size_cluster > main_cluster_size:
                    cluster2D_start[cluster] = sw_in
                    cluster2D_end[cluster] = sw_fin
                main_cluster_size = size_cluster # update the largest cluster size

        return cluster2D_start, cluster2D_end


    def _get_cluster2D_edges(self, min_cluster_size):
        """
        get the statistics of the clusters
        """
        if min_cluster_size is None:
            min_cluster_size = 0
        #cluster2D_end gives the end time of the cluster
        #cluster2D_start gives the start time of the cluster
        cluster2D_end = np.copy(self.switch2D)
        switches = np.unique(cluster2D_end)[1:]
        for sw0 in switches:
            q = cluster2D_end == sw0
            sw_next = sw0 + 1
            clusters, n_cluster = mahotas.label(q, self.NNstructure)
            for i in range(1, n_cluster+1):
                cluster = clusters == i
                size_cluster = np.sum(cluster)
                if size_cluster < min_cluster_size:
                    continue
                cluster_edge = pixels_at_edges(cluster)
                # Get the values of the switches around the cluster
                switches_at_edge = np.extract(cluster_edge, cluster2D_end)
                # Check if any of the neighbours is sw0 + 1
                # and set the original cluster to sw0 + 1
                if sw_next in switches_at_edge:
                    cluster2D_end[cluster] = sw_next
        
        cluster_switches = np.unique(cluster2D_end)[1:]
        cluster2D_start = np.copy(self.switch2D)

        for switch in cluster_switches:
            q = cluster2D_end == switch
            clusters, n_cluster = mahotas.label(q, self.NNstructure)
            for i in range(1, n_cluster+1):
                cluster = clusters == i
                cluster2D_start[cluster] = np.min(self.switch2D[cluster]) 

        return cluster2D_start, cluster2D_end


    # def _get_clusters_size_and_duration(self, clusters):
    #     # Calculus of the cluster sizes and durations
    #     print("Calculation cluster sizes and durations")
    #     cluster_sizes = np.array([])
    #     cluster_durations = np.array([])
    #     cluster_switches = np.unique(clusters)[1:]
    #     for switch in cluster_switches:
    
    #         clusters, n_cluster = mahotas.label(clusters == switch, self.NNstructure)
    #         sizes = mahotas.labeled.labeled_size(clusters)[1:]
    #         durations = np.array([np.max(np.extract(clusters == i, self.switch2D)) - switch for i in range(1, n_cluster+1)])
    #         cluster_sizes = np.concatenate((cluster_sizes, sizes))
    #         cluster_durations = np.concatenate((cluster_durations, durations))
    #     assert len(cluster_sizes) == len(cluster_durations)
    #     return cluster_sizes, cluster_durations

    # def _get_average_clusters(self, cluster_sizes, cluster_durations):
    #     # Calculus of the average cluster sizes and durations
    #     average_cluster_durations = np.unique(cluster_durations)
    #     average_cluster_sizes = []
    #     for duration in average_cluster_durations:
    #         size = np.mean(np.extract(cluster_durations == duration, cluster_sizes))
    #         average_cluster_sizes.append(size)
    #     average_cluster_sizes = np.array(average_cluster_sizes)
    #     return average_cluster_sizes, average_cluster_durations


    


#############################################################################

if __name__ == "__main__":
    try:
        mtype = sys.argv[1]
    except:
        mtype = 'Irr16'

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

    #filenamepkl = "switchTimes2D.pkl"
    #filename = "switch2D_test.pkl"
    if mtype == "Irr_800":
        rootDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/02_Irr_800uC_0.116A"
        #subDir = "NonIrr_0.095A_3s"
        #experiments = (1,2,4)
    elif mtype == "Irr16":
        rootDir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC_16e8He+/"
        subDir = "Irr_16e8He+_0.116A_3s"
        experiments = (2,3,4,5,6,7,8,9,10)
    #events_sizes = np.array([])
    #cluster_sizes = np.array([])
    #cluster_durations = np.array([])
    
    filename = os.path.join(rootDir,'switchMap2D.pkl')

    with open(filename, 'rb') as f:
        switch2D = pickle.load(f)

    events = EventsAndClusters(switch2D)    
    clusters_start, clusters_end = events.get_cluster2D(method='edges')

    # for i in experiments:
    #     sub_Dir = "%s_%s" % (str(i).rjust(2,"0"), subDir)
    #     filename = os.path.join(rootDir, sub_Dir, filenamepkl)
    #     events = Events(filename, NNstructure=NNstructure)
    #     events.get_events_and_clusters()
    #     events.plot_maps(title=sub_Dir)
    #     events_sizes = np.concatenate((events_sizes, events.events_sizes))
    #     cluster_sizes = np.concatenate((cluster_sizes, events.cluster_sizes))
    #     cluster_durations = np.concatenate((cluster_durations, events.cluster_durations))

    # average_cluster_sizes, average_cluster_durations = events._get_average_clusters(cluster_sizes, cluster_durations)
    # title = "%s%s" % (str(experiments), subDir)
    # plotEvents = PlotEvents(title=title)
    # plotEvents.size_distributions(events_sizes, events.cluster_sizes)
    # plotEvents.average_size_vs_duration_of_clusters(average_cluster_durations, average_cluster_sizes)
    #
    # This is a test
    #
