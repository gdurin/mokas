import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from visualBarkh import StackImages
import mahotas
from skimage import measure
import getLogDistributions as gLD
import mokas_events as mke
import mokas_cluster_methods as mcm
import mokas_cluster_distributions as mcd


class Bubbles(StackImages):
    """
    define a proper class to handle
    the sequence of images
    taken from bubbles
    """
    def __init__(self, **imParameters):
        NN = 3
        self.NNstructure = np.ones((NN,NN))

        StackImages.__init__(self, **imParameters)


    def switches(self): 
        return np.unique(self._switchTimes2D)[1:]
           

    def getEventsAndClusters(self, data=None, method='edges'):
        self.get_events_and_clusters = mke.EventsAndClusters(self._switchTimes2D, NNstructure=self.NNstructure)
        out = self.get_events_and_clusters.get_cluster2D(method='edges')
        self.cluster2D_start, self.cluster2D_end = out


    def plotEventsAndClustersMaps(self):
        try:
            q = self.cluster2D_start
        except:
            print("Run getEventsAndClusters first!")

        clrs = (np.random.rand(2*len(self.switches()),3) + [1,1,1])/2
        clrs[0] = [0,0,0]
        cmap = mpl.colors.ListedColormap(clrs)
        
        cluster_switches = np.unique(self.cluster2D_start)[1:]

        # Plot
       
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True) # ColorImages of events and clusters
          
        axs[0].imshow(self._switchTimes2D, cmap=cmap)
        axs[1].imshow(self.cluster2D_start, cmap=cmap)
        axs[2].imshow(self.cluster2D_end, cmap=cmap)

        axs[0].set_title('Events')
        axs[1].set_title('Clusters start')
        axs[2].set_title('Clusters end')

        fig.suptitle('Events and clusters', fontsize=20)

        rows, cols = self._switchTimes2D.shape
        axs[0].axis((0,cols,rows,0))
        font = {'weight': 'normal', 'size': 8}
        for i in cluster_switches:
            cluster = self.cluster2D_start == i
            cnts = measure.find_contours(cluster, 0.5, fully_connected='high')
            for cnt in cnts:
                X,Y = cnt[:,1], cnt[:,0]
                for ax in [axs[0], axs[1], axs[2]]:
                    ax.plot(X, Y, c='k', antialiased=True, lw=1)






        

