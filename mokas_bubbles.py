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
import configparser



class Bubbles_ini(object):
    def __init__(self, filepath):
        self.imParameters = dict()
        self.config = configparser.ConfigParser()
        filename = os.path.join(filepath, "bubbles.ini")
        if not os.path.isfile(filename):
            print("Please, prepare a bubbles.ini file")
        else:
            print(filename)
        
        self.config.read(filename)
        self.default = self.config['DEFAULT']
        self.filename_suffix = self.default['filename_suffix']
        self.imParameters['firstIm'] = int(self.default['firstIm'])
        self.imParameters['lastIm'] = int(self.default['lastIm'])
        self.imParameters['filtering'] = self.default['filtering']
        self.imParameters['sigma'] = float(self.default['sigma'])
        self.imParameters['hdf5_use'] = self.default['hdf5'] == 'False'
        if self.imParameters['hdf5_use']:
            user = self.default['user']
            self.imParameters['hdf5_signature'] = {'user': user}
        
        bubble = self.config['bubble']
        crop_upper_left_pixel = tuple([int(n) for n in bubble['crop_upper_left_pixel'].split(",")])
        crop_lower_right_pixel = tuple([int(n) for n in bubble['crop_lower_right_pixel'].split(",")])
        self.imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        self.experiments = [int(n) for n in bubble['experiments'].split(",")]

        analysis = self.config['analysis']
        self.thresholds = [float(n) for n in analysis['thresholds'].split(",")]



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


    def plotEventsAndClustersMaps(self, fig=None, axs=None):
        try:
            q = self.cluster2D_start
        except:
            print("Run getEventsAndClusters first!")

        clrs = np.random.rand(2*len(self.switches()),3)
        clrs[0] = [0,0,0]
        cmap = mpl.colors.ListedColormap(clrs)
        
        cluster_switches = np.unique(self.cluster2D_start)[1:]

        # Plot

        if not fig:
            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True) # ColorImages of events and clusters
            ax0, ax1, ax2 = axs[0], axs[1], axs[2]
        else:
            ax0, ax1, ax2 = axs
           
        ax0.imshow(self._switchTimes2D, cmap=cmap)
        ax1.imshow(self.cluster2D_start, cmap=cmap)
        ax2.imshow(self.cluster2D_end, cmap=cmap)

        ax0.set_title('Events')
        ax1.set_title('Clusters start')
        ax2.set_title('Clusters end')

        fig.suptitle('Events and clusters', fontsize=20)

        rows, cols = self._switchTimes2D.shape
        ax0.axis((0,cols,rows,0))
        font = {'weight': 'normal', 'size': 8}
        for i in cluster_switches:
            cluster = self.cluster2D_start == i
            cnts = measure.find_contours(cluster, 0.5, fully_connected='high')
            for cnt in cnts:
                X,Y = cnt[:,1], cnt[:,0]
                for ax in [ax0, ax1, ax2]:
                    ax.plot(X, Y, c='k', antialiased=True, lw=1)






        

