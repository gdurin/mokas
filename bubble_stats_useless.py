import sys, os
import cPickle as pickle
import numpy as np
import mahotas
import getLogDistributions as gLD
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import measure
import mokas_events as mke

def events_and_clusters(self, data=None, is_plot=True):
    NNstructure = np.ones((3,3))
    
    sw = np.unique(switch2D)[1:]
    # statistics of events
    events_sizes = np.array([])

    for switch in sw:
        q = switch2D == switch
        im, n_cluster = mahotas.label(q, NNstructure)
        sizes = mahotas.labeled.labeled_size(im)[1:]
        events_sizes = np.concatenate((events_sizes, sizes))

    # statistics of clusters
    #avalanche_end gives the end of the cluster while aavalanche_start gives the beginning
    print("Calculating clusters")
    progress = "."
    avalanches_end = np.copy(switch2D)

    for sw0 in sw:
        q = avalanches_end == sw0
        clusters, n_cluster = mahotas.label(q, NNstructure)
        for i in range(1, n_cluster+1):
            cluster = clusters == i
            cluster_edge = mke.pixels_at_edges(cluster)
            # Get the values of the switches around the cluster
            switches_at_edge = np.extract(cluster_edge, avalanches_end)
            sw_next = sw0 + 1
            # Check if any of the neighbours is sw0 + istep
            # and set the original cluster to sw0 + istep
            if sw_next in switches_at_edge:
                avalanches_end[cluster] = sw_next


    avalanche_switches = np.unique(avalanches_end)[1:]
    avalanches_start = np.copy(switch2D)

    for switch in avalanche_switches:
        q = avalanches_end == switch
        clusters, n_cluster = mahotas.label(q, NNstructure)
        for i in range(1, n_cluster+1):
            cluster = clusters == i
            avalanches_start[cluster] = np.min(switch2D[cluster]) 




    # Calculus of the cluster sizes and durations
    print("Calculating cluster sizes and durations")
    avalanche_sizes = np.array([])
    avalanche_durations = []
    for switch in avalanche_switches:
        clusters, n_cluster = mahotas.label(avalanches_end == switch, NNstructure)
        sizes = mahotas.labeled.labeled_size(clusters)[1:]
        avalanche_sizes = np.concatenate((avalanche_sizes, sizes))
        for i in range(1, n_cluster+1):
            q = clusters == i
            d = avalanches_end[q][0] - avalanches_start[q][0]
            avalanche_durations.append(d)
    avalanche_durations = np.asarray(avalanche_durations)
    assert len(avalanche_sizes) == len(avalanche_durations)
    
    average_avalanche_durations = np.unique(avalanche_durations)
    average_avalanche_sizes = []
    for duration in average_avalanche_durations:
        size = np.mean(np.extract(avalanche_durations == duration, avalanche_sizes))
        average_avalanche_sizes.append(size)
    average_avalanche_sizes = np.array(average_avalanche_sizes)

    if is_plot:
        clrs = (np.random.rand(2*len(sw),3) + [1,1,1])/2
        clrs[0] = [0,0,0]
        cmap = mpl.colors.ListedColormap(clrs)

        fig1, axs1 = plt.subplots(1, 3, sharex=True, sharey=True) # ColorImages of events and sizes
        axs1[0].imshow(switch2D, cmap=cmap)
        axs1[1].imshow(avalanches_start, cmap=cmap)
        axs1[2].imshow(avalanches_end, cmap=cmap)

        for i in avalanche_switches:
            cluster = avalanches_end==i
            cnts = measure.find_contours(cluster, 0.5, fully_connected = 'high')
            for cnt in cnts:
                X,Y = cnt[:,1], cnt[:,0]
                for ax in axs1:
                    ax.plot(X, Y, c='k', antialiased=True, lw=1)

        print("We have collected %i events and %i avalanches" % (len(events_sizes), len(avalanche_sizes)))

        fig2, axs2 = plt.subplots(1, 2) # Distributions of events and avalanches
        # Calculate and plot the distributions of clusters and avalanches
        for sizes, label in zip([events_sizes, avalanche_sizes], ['events', 'avalanches_end']):
            x, y, yerr = gLD.logDistribution(sizes, log_step=0.1, 
                                   first_point=1., normed=True)
            # Plots of the distributions
            axs2[0].loglog(x, y,'o', label=label)
            axs2[0].errorbar(x, y, yerr, fmt="none")
            if label == 'events':
                axs2[0].loglog(x, 0.4 * x**-1.17 * np.exp(-x/50),'-', label=r'S^{-1.17} exp(-S/50)')
            elif label == 'avalanches_end':
                axs2[0].loglog(x, 0.4 * x**-1.17 * np.exp(-x/500),'-', label=r'S^{-1.17}')
        axs2[0].legend()
        axs2[0].grid(True)

        axs2[1].loglog(average_avalanche_durations, average_avalanche_sizes,'o')
        axs2[1].grid(True)
        plt.show()
    return avalanches_start, avalanches_end



if __name__ == "__main__":
    qq = np.array([[13, 13, 13,  8,  8,  3],
       [12, 14,  8,  6,  3,  3],
       [ 9,  4,  4,  4,  3, -1],
       [15, 15, 15,  2, -1, -1],
       [15, 4, -1, -1, -1, -1],
       [-1, -1, -1, -1, -1, -1]], dtype=np.int32)
    
    qq2 = np.array([[13, 13, 13,  8,  7,  3],
       [12, 14,  8,  6,  3,  3],
       [ 8,  4,  4,  4,  3, -1],
       [15,  9, 15,  2, -1, -1],
       [15, 15,  2, -1, -1, -1],
       [-1, 2, -1, -1, -1, -1]], dtype=np.int32)
    if False:
        print("Loading pickle file")
        print(sys.argv)
        try:
            choice = sys.argv[1]
        except:
            choice = 'bubble'
        if choice == "bubble":
            k = sys.argv[2]
            print k
            k = str(k).rjust(2,"0")
            rootDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/%s_Irr_800uC_0.116A" % k
            filename = os.path.join(rootDir,'switchMap2D.pkl')

            with open(filename, 'rb') as f:
                switch2D = pickle.load(f)
    else:
        switch2D = qq
    avalanches_start, avalanches_end = events_and_clusters(switch2D)
    #print(qq2)
    #print(30*"#")
    #print(avalanches)