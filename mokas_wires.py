import os, glob
import math
import configparser
import numpy as np
import matplotlib.pyplot as plt
from visualBarkh import StackImages
import mahotas
import skimage.morphology as morph
import skimage.feature as feature
from skimage import measure
import mokas_events as mke
import mokas_cluster_methods as cmet


class Wires_ini(object):
    def __init__(self, filepath, n_wire=1):
        self.imParameters = dict()
        self.config = configparser.ConfigParser()
        filename = os.path.join(filepath, "wires.ini")
        if not os.path.isfile(filename):
            print("Please, prepare a wires.ini file")
        else:
            print(filename)
        self.config.read(filename)
        self.default = self.config['DEFAULT']
        self.n_wires = int(self.default['n_wires'])
        self.filename_suffix = self.default['filename_suffix']
        self.imParameters['firstIm'] = int(self.default['firstIm'])
        self.imParameters['lastIm'] = int(self.default['lastIm'])
        self.imParameters['filtering'] = self.default['filtering']
        self.imParameters['sigma'] = float(self.default['sigma'])
        self.motion = self.default['motion']
        self.imParameters['rotation'] = float(self.default['rotation'])
        self.edge_trim_percent = int(self.default['edge_trim_percent'])
        self.imParameters['hdf5_use'] = self.default['hdf5'] == 'True'
        if self.imParameters['hdf5_use']:
            user = self.default['user']
            self.imParameters['hdf5_signature'] = {'n_wire' : "wire%i" % n_wire, 'user': user}
        if n_wire > self.n_wires:
            print("Number of the wire not available (1-%i)" % self.n_wires)
        nwire = "wire%i" % n_wire
        nw = self.config[nwire]
        crop_upper_left_pixel = tuple([int(n) for n in nw['crop_upper_left_pixel'].split(",")])
        crop_lower_right_pixel = tuple([int(n) for n in nw['crop_lower_right_pixel'].split(",")])
        self.imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        self.experiments = [int(n) for n in nw['experiments'].split(",")]
        analysis = self.config['Analysis']
        self.threshold = float(analysis['threshold'])


class Wires(StackImages):
    """
    define a proper class to handle
    the sequence of images
    taken from wires

    Parameters:
    motion : has to be downward. will be erased in future versions
    calculate_edges : trim the edges of the wires
    zoom_in_data : False|True
        (do not) show the portion of the wire where the motion occurs
    """
    def __init__(self, motion='downward', edge_trim_percent=None,
        zoom_in_data=True, **imParameters):
        imParameters['exclude_switches_from_central_domain'] = False # Do not change
        # By default we get rid of the switched calculated out of the final domain
        imParameters['exclude_switches_out_of_final_domain'] = True
        StackImages.__init__(self, **imParameters)
        self.motion = motion
        self.rotation = imParameters['rotation']
        self.zoom_in_data = zoom_in_data
        n, rows, cols = self.shape
        # if motion == 'downward':
        #     self.ref_point = (0, cols//2)
        # elif motion == 'upward':
        #     self.ref_point = (rows, cols//2)
        # elif motion == 'leftward':
        #     self.ref_point = (rows//2, cols)
        # elif motion == 'rightward':
        #     self.ref_point = (rows//2, 0)
        # All the figure have to be roated to get the motion downward
        self.ref_point = (0, cols//2)
        self.edge_trim_percent = edge_trim_percent
        if edge_trim_percent:
            print("Trim edges...")
            self.row_profile = np.mean(self.Array[0], 0)
            p1, p2 = self._get_edges(self.Array[0])
            self.Array = self.Array[:, :, p1:p2+1]
            n, self.dimX, self.dimY = self.Array.shape
        # For wires, a larger NN is required for cluster detection
        NN = 5
        self.NNstructure = np.ones((NN,NN))

    @property
    def switches(self): 
        return np.unique(self._switchTimes2D)[1:]

    def get_stats_prop(self, min_size=30):
        """
        calculate the statistical properties 
        of the avalanches
        """
        print("Calculating the statistical properties of avalanches")
        # Calculation of sizes
        self.sizes_whole = np.array([sum(self._switchTimes2D == sw) for sw in self.switches])
        # Here we need to get the largest cluster and get its properties
        self.stats_prop = dict()
        sizes = []
        lenghts_initial = []
        curvatures_initial = []
        lenghts_final = []
        curvatures_final = []
        sws = []
        #image_corners = np.zeros_like(self._switchTimes2D).astype('bool')
        for i, sw in enumerate(self.switches):
            #print(i, sw)
            im = self._switchTimes2D == sw
            largest_cluster, cluster_size = self._largest_cluster(im)
            if cluster_size >= min_size:
                out = self._get_upper_and_lower_contour(largest_cluster, 
                    is_largest_size_only=False)
                l_initial, l_final, L_linear, success = out
                if success:
                    if len(l_initial) == 0 or len(l_final) == 0: # in case of errors
                        print("Error for switch: %i, iteration %i" % (sw, i))
                        next
                    length, curvature = self._get_lenght_and_curvature(l_initial, curvature=True)
                    if length is not None:
                        lenghts_initial.append(length)
                        length, curvature = self._get_lenght_and_curvature(l_final, curvature=True)
                        lenghts_final.append(length)
                        curvatures_final.append(curvature)
                        sws.append(sw)
                        sizes.append(cluster_size)
                #image_corners = image_corners + i * im_corners.astype('bool')
        self.stats_prop['sizes'] = np.array(sizes)
        self.stats_prop['lenghts_initial'] = np.array(lenghts_initial)
        self.stats_prop['lenghts_final'] = np.array(lenghts_final)
        self.stats_prop['curvatures_initial'] = np.array(curvatures_initial)
        self.stats_prop['curvatures_final'] = np.array(curvatures_final)
        #self.stats_prop['image_corners'] = image_corners
        self.switches_above_min_size = np.array(sws)
        print("Done.")

    def _get_edges(self, im):
        """
        Calculate the position of the edge
        from the first row image
        edge_trim_percent reduces the width of the wire
        from both sides
        """
        gray_profile = np.mean(im, 0)
        L2 = len(gray_profile) / 2
        p1 = np.argmin(gray_profile[:L2])
        p2 = np.argmin(gray_profile[L2:]) + L2
        distance = p2 - p1
        p1 += distance * self.edge_trim_percent / 100
        p2 -= distance * self.edge_trim_percent / 100
        # out_mean1 = np.mean(gray_profile[:L2/5])
        # out_mean2 = np.mean(gray_profile[-L2/5:])
        # p1 += np.argmax(gray_profile[p1:L2] - out_mean1 > 0) 
        # gp = gray_profile[L2:p2 + 1] - out_mean2 > 0
        # p2 -= np.argmax(gp[::-1])
        return p1, p2

    def _get_lenght_and_curvature(self, line):
        return cmet.get_lenght_and_curvature(line)

    def _find_corners(self, cluster, n_fast=12, threshold_fast=0.1, method='farthest'):
        return cmet.find_corners(cluster, n_fast=n_fast, threshold_fast=threshold_fast, method=method)

    def _get_upper_and_lower_contour(self, cluster, n_fast=12, threshold_fast=0.1, 
        is_largest_size_only=True, test=False):
        return cmet.get_upper_and_lower_contour(cluster, self.motion, self.ref_point, 
            n_fast, threshold_fast, is_largest_size_only, test)
        
    def _largest_cluster(self, im, NNstructure=None):
        if not NNstructure:
            NNstructure = self.NNstructure
        return cmet.largest_cluster(im, NNstructure)

    def _sizes_largest_clusters(self):
        sizes = np.zeros_like(self.imageNumbers)
        for switch in self.switches:
            cluster = self._switchTimes2D == switch
            cluster, cluster_size = self._largest_cluster(cluster)
            sizes[switch] = cluster_size
        return sizes

    def find_contours(self, lines_color=None, invert_y_axis=True, step_image=1,
                        consider_events_around_a_central_domain=False, 
                        initial_domain_region=None, remove_bordering=False,
                        plot_centers_of_mass = False, reference=None, 
                        rescale_area=False, plot_rays=True,
                        fig=None, ax=None, title=None):
        if fig is None:
            fig = plt.figure(figsize=self._figColorImage.get_size_inches())
            ax = fig.gca()
        else:
            plt.figure(fig.number)
            if ax is None:
                ax = fig.gca()
        print("Print contours....")
        self.contours = {}
        switch0 = self.switches[0]
        cluster = self._switchTimes2D == switch0
        for switch in self.switches[1:]:
            cluster += self._switchTimes2D == switch
            cluster, cluster_size = self._largest_cluster(cluster)
            cnts_all = measure.find_contours(cluster, 0.5)
            cnts_all = self._find_longest_contours(cnts_all, 2)
            self.contours[switch] = cnts_all
            for cnts in cnts_all:
                X, Y = cnts[:,1], cnts[:,0]
                ax.plot(X, Y, c='k', antialiased=True, lw=1)
        self.is_find_contours = True
        if invert_y_axis:
            ax.invert_yaxis()
        plt.show()

    def _find_longest_contours(self, cnts, n_contours=2):
        """
        choose the "n_contours" largest contours
        """
        lengths = [len(cnt) for cnt in cnts]
        if len(lengths) > n_contours:
            out = []
            for i in range(n_contours):
                i0 = np.argmax(lengths)
                out.append(cnts[i0])
                lengths.pop(i0)
                cnts.pop(i0)
            return out
        else:
            return cnts

    def _zeros(self, threshold, method='sub_cluster'):
        """
        Find the zeros of the histogram
        i.e. where the signal == threshold
        Parameters:
        method : srt
            full_histogram :  signal are the values of the histo
            sub_cluster : signal are the values of the largest clusters only

        """
        if method == 'full_histogram':
            signal = self.N_hist
        elif method == 'sub_cluster':
            signal = self._sizes_largest_clusters()
        # Find the avalanche zeros
        # 1. step function for v=r
        fv = np.where(signal > threshold, 1, 0)
        # 2. Derivative
        # +1 :"index of the beginning of the avalanche"
        # -1 :"index of end the of the avalanche -1"
        dfv = np.diff(fv)
        # Check that the first nonzero value must be
        # 1 and the last -1; get rid of the uncorrect values
        nonzeroIndex = np.nonzero(dfv)[0]
        if dfv[nonzeroIndex[0]] == -1:
            nonzeroIndex = nonzeroIndex[1:]
        if dfv[nonzeroIndex[-1]] == 1:
            nonzeroIndex = nonzeroIndex[:-1]

        # check if evaluation is correct: even n. of data
        if len(nonzeroIndex) % 2:
            print("Error in evaluating the avalanche limits")

        # The limits belows are calculated 
        # when the cluster is larger than the threshold
        # Array of the start of the cluster
        x0s = nonzeroIndex[::2] + 1
        # Array of the end of the cluster
        x1s = nonzeroIndex[1::2]
        if x1s[-1] > len(signal):
            x1s[-1] = len(signal)
        return x0s, x1s

    def plotEventsAndClusters(self, cluster_threshold=5, method='sub_cluster', 
                                fig=None, axs=None, title=None, with_cluster_number=True):
        """
        method: str
            sub_cluster: detect if there is a sub_cluster larger than the threshold
            full_histogram: detect if there total number of switches is larger than the threshold
        """
        if not self.is_histogram:   
            self.plotHistogram(self._switchTimes2D)
        x0s, x1s = self._zeros(cluster_threshold, method=method)
        self.events_and_clusters = mke.EventsAndClusters(self._switchTimes2D, NNstructure=self.NNstructure)
        self.events_and_clusters.get_events_and_clusters(cluster_limits=zip(x0s,x1s))
        self.events_and_clusters.plot_maps(self._colorMap, zoom_in_data=self.zoom_in_data, 
                                            fig=fig, axs=axs, title=title, with_cluster_number=False)


    def post_processing(self, compare_to_row_images=False, fillValue=-1):
        """
        This is an experimental feature to get rid of
        (small) sub_clusters which do not have a corresponding significant
        variation in the gray scale of the row images
        """
        switch2D = np.copy(self._switchTimes2D)
        if compare_to_row_images:
            row_data2D = self.Array
            for sw in self.switches:
                q = switch2D == sw
                sub_clusters, n_sub_clusters = mahotas.label(q, self.NNstructure)
                for i in range(1, n_sub_clusters+1):
                    p = sub_clusters == i
                    average_gray_step = np.mean(row_data2D[sw,p]-row_data2D[sw-1,p])
                    print(average_gray_step)
                    if np.abs(average_gray_step) < self._threshold/2.:
                        switch2D[p] = fillValue
                        print("Done")
        return switch2D

