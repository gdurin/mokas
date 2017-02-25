import sys
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

def test(switch2D, is_plot=False, set_init_time=True):
    NNstructure = np.ones((3,3))
    
    sw = np.unique(switch2D)[1:]
    # statistics of events
    events_sizes = np.array([])

    for switch in sw:
        q = switch2D == switch
        im, n_cluster = mahotas.label(q, NNstructure)
        sizes = mahotas.labeled.labeled_size(im)[1:]
        events_sizes = np.concatenate((events_sizes, sizes))

    # statistics of avalanches
    print("Calculation of avalanches")
    progress = "."
    avalanches = np.copy(switch2D)
    if set_init_time:
        istep = -1
    else:
        istep = 1
    for sw0 in sw[::istep]:
        q = avalanches == sw0
        clusters, n_cluster = mahotas.label(q, NNstructure)
        for i in range(1, n_cluster+1):
            cluster = clusters == i
            cluster_edge = pixels_at_edges(cluster)
            # Get the values of the switches around the cluster
            switches_at_edge = np.extract(cluster_edge, avalanches)
            sw_next = sw0 + istep
            # Check if any of the neighbours is sw0 + istep
            # and set the original cluster to sw0 + istep
            if sw_next in switches_at_edge:
                avalanches[cluster] = sw_next

    # Calculus of the avalanche sizes and durations
    print("Calculation avalanche sizes and durations")
    avalanche_sizes = np.array([])
    avalanche_durations = np.array([])
    avalanche_switches = np.unique(avalanches)[1:]
    for switch in avalanche_switches:
        clusters, n_cluster = mahotas.label(avalanches == switch, NNstructure)
        sizes = mahotas.labeled.labeled_size(clusters)[1:]
        durations = np.array([np.max(np.extract(clusters == i, switch2D)) - switch for i in range(1, n_cluster+1)])
        avalanche_sizes = np.concatenate((avalanche_sizes, sizes))
        avalanche_durations = np.concatenate((avalanche_durations, durations))
    assert len(avalanche_sizes) == len(avalanche_durations)
    average_avalanche_durations = np.unique(avalanche_durations)
    average_avalanche_sizes = []
    for duration in average_avalanche_durations:
        size = np.mean(np.extract(avalanche_durations == duration, avalanche_sizes))
        average_avalanche_sizes.append(size)
    average_avalanche_sizes = np.array(average_avalanche_sizes)

    if is_plot:
        clrs = (np.random.rand(256,3) + [1,1,1])/2
        clrs[0] = [0,0,0]
        cmap = mpl.colors.ListedColormap(clrs)

        fig1, axs1 = plt.subplots(1, 2, sharex=True, sharey=True) # ColorImages of events and sizes
        axs1[0].imshow(switch2D, cmap=cmap)
        axs1[1].imshow(avalanches, cmap=cmap)
        for i in avalanche_switches:
            cluster = avalanches==i
            cnts = measure.find_contours(cluster, 0.5)
            for cnt in cnts:
                X,Y = cnt[:,1], cnt[:,0]
                for ax in axs1:
                    ax.plot(X, Y, c='k', antialiased=True, lw=1)

        print("We have collected %i events and %i avalanches" % (len(events_sizes), len(avalanche_sizes)))

        fig2, axs2 = plt.subplots(1, 2) # Distributions of events and avalanches
        # Calculate and plot the distributions of clusters and avalanches
        for sizes, label in zip([events_sizes, avalanche_sizes], ['events', 'avalanches']):
            x, y, yerr = gLD.logDistribution(sizes, log_step=0.1, 
                                   first_point=1., normed=True)
            # Plots of the distributions
            axs2[0].loglog(x, y,'o', label=label)
            axs2[0].errorbar(x, y, yerr, fmt="none")
            if label == 'events':
                axs2[0].loglog(x, 0.4 * x**-1.17 * np.exp(-x/50),'-', label=r'S^{-1.17} exp(-S/50)')
            elif label == 'avalanches':
                axs2[0].loglog(x, 0.4 * x**-1.17 * np.exp(-x/500),'-', label=r'S^{-1.17}')
        axs2[0].legend()
        axs2[0].grid(True)

        axs2[1].loglog(average_avalanche_durations, average_avalanche_sizes,'o')
        axs2[1].grid(True)
        plt.show()
    return avalanches

if __name__ == "__main__":
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
    if True:
        print("File loading")
        filename = "switch2D_05.pkl"
        with open(filename, 'rb') as f:
            switch2D = pickle.load(f)
    else:
        switch2D = qq2
    avalanches = test(switch2D, is_plot=True)
    #print(qq2)
    #print(30*"#")
    #print(avalanches)