import sys, os
import numpy as np
import matplotlib.pyplot as plt
from visualBarkh import StackImages
import mahotas
from skimage import measure
import getLogDistributions as gLD
import mokas_events as mke
import mokas_cluster_methods as mcm
import mokas_cluster_distribution as mcd


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
        
        

    def get_events_and_clusters(self, data=None, method='edges'):
        if data is None:
            data = self._switchTimes2D

        

