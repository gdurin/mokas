import sys, os
import pickle
import numpy as np
import mahotas
import getLogDistributions as gLD
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import measure


def pixels_at_edges(cluster, with_diagonal=True):
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
            rolled = np.roll(cluster, shift, axis=0)
            index = shift - 1 * (shift==1)
            rolled[index, :] = False
            for shift2 in [-1,1]:
                rolled = np.roll(rolled, shift2, axis=1)
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
    return z

class Events:
    def __init__(self, switch2D, set_init_time=True, NNstructure=None):
        """
        switch2D is the map of switches
        or the filename
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
            self.NNstructure = np.ones((3,3))
        else:
            self.NNstructure = NNstructure
        self.switches = np.unique(self.switch2D)[1:]

    def get_events_and_avalanches(self):
        print("Calculation of events")
        self.events_sizes = self._get_events()
        print(len(self.events_sizes))
        print("Calculation of avalanches")
        self.avalanche2D = self._get_avalanche2D()
        self.avalanche_switches = np.unique(self.avalanche2D)[1:]
        size_and_durations = self._get_avalanches(self.avalanche2D)
        self.avalanche_sizes, self.avalanche_durations = size_and_durations
        print("Calculation of average avalanches")
        averages = self._get_average_avalanches(*size_and_durations)
        self.average_avalanche_sizes, self.average_avalanche_durations = averages
        print("We have collected %i events and %i avalanches" % (len(self.events_sizes), len(self.avalanche_sizes)))

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

    def _get_avalanche2D(self):
        # statistics of avalanches
        avalanche2D = np.copy(self.switch2D)
        if self.set_init_time:
            istep = -1
        else:
            istep = 1
        for sw0 in self.switches[::istep]:
            q = avalanche2D == sw0
            clusters, n_cluster = mahotas.label(q, self.NNstructure)
            for i in range(1, n_cluster+1):
                cluster = clusters == i
                cluster_edge = pixels_at_edges(cluster)
                # Get the values of the switches around the cluster
                switches_at_edge = np.extract(cluster_edge, avalanche2D)
                sw_next = sw0 + istep
                # Check if any of the neighbours is sw0 + istep
                # and set the original cluster to sw0 + istep
                if sw_next in switches_at_edge:
                    avalanche2D[cluster] = sw_next
        return avalanche2D

    def _get_avalanches(self, avalanches):
        # Calculus of the avalanche sizes and durations
        print("Calculation avalanche sizes and durations")
        avalanche_sizes = np.array([])
        avalanche_durations = np.array([])
        avalanche_switches = np.unique(avalanches)[1:]
        for switch in avalanche_switches:
            clusters, n_cluster = mahotas.label(avalanches == switch, NNstructure)
            sizes = mahotas.labeled.labeled_size(clusters)[1:]
            durations = np.array([np.max(np.extract(clusters == i, self.switch2D)) - switch for i in range(1, n_cluster+1)])
            avalanche_sizes = np.concatenate((avalanche_sizes, sizes))
            avalanche_durations = np.concatenate((avalanche_durations, durations))
        assert len(avalanche_sizes) == len(avalanche_durations)
        return avalanche_sizes, avalanche_durations

    def _get_average_avalanches(self, avalanche_sizes, avalanche_durations):
        # Calculus of the avalanche sizes and durations
        average_avalanche_durations = np.unique(avalanche_durations)
        average_avalanche_sizes = []
        for duration in average_avalanche_durations:
            size = np.mean(np.extract(avalanche_durations == duration, avalanche_sizes))
            average_avalanche_sizes.append(size)
        average_avalanche_sizes = np.array(average_avalanche_sizes)
        return average_avalanche_sizes, average_avalanche_durations

    def plot_maps(self, cmap='pastel', title=None):
        if cmap == 'pastel':
            clrs = (np.random.rand(256,3) + [1,1,1])/2
            clrs[0] = [0,0,0]
            cmap = mpl.colors.ListedColormap(clrs)

        fig1, axs1 = plt.subplots(1, 2, sharex=True, sharey=True) # ColorImages of events and sizes
        axs1[0].imshow(self.switch2D, cmap=cmap)
        axs1[1].imshow(self.avalanche2D, cmap=cmap)
        for i in self.avalanche_switches:
            cluster = self.avalanche2D==i
            cnts = measure.find_contours(cluster, 0.5)
            for cnt in cnts:
                X,Y = cnt[:,1], cnt[:,0]
                for ax in axs1:
                    ax.plot(X, Y, c='k', antialiased=True, lw=1)
        fig1.suptitle(title,fontsize=30)


class PlotEvents:
    def __init__(self, fig=None, axs=None, title=None):
        self.title = title

    def size_distributions(self, events_sizes, avalanche_sizes):
        fig, axs = plt.subplots(1, 1) # Distributions of events and avalanches
        fig.suptitle(self.title,fontsize=30)
        # Calculate and plot the distributions of clusters and avalanches
        for sizes, label in zip([events_sizes, avalanche_sizes], ['events', 'avalanches']):
            x, y, yerr = gLD.logDistribution(sizes, log_step=0.1, 
                                   first_point=1., normed=True)
            # Plots of the distributions
            axs.loglog(x, y,'o', label=label)
            axs.errorbar(x, y, yerr, fmt=None)
            if label == 'events':
                axs.loglog(x, 0.25 * x**-1.17 * np.exp(-x/100),'-', label=r'S^{-1.17} exp(-S/50)')
            elif label == 'avalanches':
                axs.loglog(x, 0.25 * x**-1.17,'-', label=r'S^{-1.17} exp(-S/50)')
        axs.legend()
        axs.grid(True)

    def average_size_vs_duration_of_avalanches(self, durations, sizes):
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
    avalanche_sizes = np.array([])
    avalanche_durations = np.array([])
    
    for i in experiments:
        sub_Dir = "%s_%s" % (str(i).rjust(2,"0"), subDir)
        filename = os.path.join(rootDir, sub_Dir, filenamepkl)
        events = Events(filename, NNstructure=NNstructure)
        events.get_events_and_avalanches()
        events.plot_maps(title=sub_Dir)
        events_sizes = np.concatenate((events_sizes, events.events_sizes))
        avalanche_sizes = np.concatenate((avalanche_sizes, events.avalanche_sizes))
        avalanche_durations = np.concatenate((avalanche_durations, events.avalanche_durations))

    average_avalanche_sizes, average_avalanche_durations = events._get_average_avalanches(avalanche_sizes, avalanche_durations)
    title = "%s%s" % (str(experiments), subDir)
    plotEvents = PlotEvents(title=title)
    plotEvents.size_distributions(events_sizes, events.avalanche_sizes)
    plotEvents.average_size_vs_duration_of_avalanches(average_avalanche_durations, average_avalanche_sizes)