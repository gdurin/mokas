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
        if n_wire > self.n_wires:
            print("Number of the wire not available (1-%i)" % self.n_wires)
        nwire = "n%i" % n_wire
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
                l_initial, l_final, im_corners = self._get_upper_and_lower_contour(largest_cluster, is_largest_size_only=False)
                if len(l_initial) == 0 or len(l_final) == 0: # in case of errors
                    print("Error for switch: %i, iteration %i" % (sw, i))
                    next
                length, curvature = self._get_lenght_and_curvature(l_initial)
                if length is not None:
                    lenghts_initial.append(length)
                    length, curvature = self._get_lenght_and_curvature(l_final)
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
        """
        Meausure the length of a contour line
        in pixel units
        """
        try:
            x, y = line[:,1], line[:,0]
            lenght = np.sum(((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)**0.5)
            curvature, b, c = np.polyfit(x, y, 2)
        except TypeError:
            return None, None
        return lenght, -curvature

    def _find_corners(self, cluster, n_fast=12, threshold_fast=0.1, method='farthest'):
        """
        find 2 corners in the cluster
        Method:
        'farthest' from the center of mass
        'largest' between all the corners
        """

        # Find the corners with the corner_fast method
        n_clusters = 0
        while n_clusters < 2:
            try:
                cf = feature.corner_fast(cluster, n_fast, threshold_fast)
                # Now find the clusters associated to them
                im, n_clusters = mahotas.label(cf)
                n_fast -= 1
            except OverflowError:
                return False, False
        # It is better to get the two corners
        if n_clusters > 2:
            if method == 'largest':
                sizes = mahotas.labeled.labeled_size(im)
                # get the two largest
                min_size = np.sort(sizes[1:])[-2:][0]
                too_small = np.where(sizes < min_size)
                im = mahotas.labeled.remove_regions(im, too_small)
            elif method == 'farthest':
                d2 = []
                r0, c0 = mahotas.center_of_mass(cluster)
                regions = np.arange(1, n_clusters+1)
                for i in regions:
                    r1, c1 = mahotas.center_of_mass(im == i)
                    distance = (r0 - r1)**2 + (c0 - c1)**2
                    d2.append(distance)
                dmin = np.sort(d2)[-2]
                is_to_remove = d2 < dmin
                regions = regions[is_to_remove]
                im = mahotas.labeled.remove_regions(im, regions)
            else:
                print("Not implemented")
                return None, False
            im, n_left = mahotas.labeled.relabel(im)
            try:
                assert n_left == 2
            except AssertionError:
                return im, False
        return im, True

    def _mean_distance(self, cnt, ref_point):
        """
        Find the mean distance between a contour
        and a reference point
        """
        y,x = np.hsplit(cnt,2)
        y0, x0 = ref_point
        mean_distance = np.mean(((y-y0)**2 + (x-x0)**2)**0.5)
        return mean_distance

    def _get_upper_and_lower_contour(self, cluster, n_fast=12, threshold_fast=0.1, 
        is_largest_size_only=True, test=False):
        """
        This is really a tough problem!
        Find the initial and final DW position 
        in a cluster
        Method:
        1. Find the corners (to be ckeck in limit cases)
        2. Find the contour of the corners and of the cluster
        3. Find the common elements of the two contous
        4. Decide the position of the corner (the middle element)
        5. Split the cluster contour in two sub-arrays
        
        
        Extract FAST corners for a given image.
        Parameters: 
        image : 2D ndarray
        n : int
        Minimum number of consecutive pixels out of 16 pixels
        on the circle that should all be either brighter or darker w.r.t testpixel.
        A point c on the circle is darker w.r.t test pixel p if Ic < Ip - threshold 
        and brighter if Ic > Ip + threshold. 
        Also stands for the n in FAST-n corner detector.
        threshold : float
        Threshold used in deciding whether the pixels on the circle are brighter, 
        darker or similar w.r.t. the test pixel. Decrease the threshold when more corners
        are desired and vice-versa.

        """
        if is_largest_size_only:
            cluster, cluster_size = self._largest_cluster(cluster)
        cluster = morph.remove_small_holes(cluster)
        cnt_cluster = measure.find_contours(cluster,0.5)[0]
        cnt_cluster_2_string = np.array(["%s,%s" % (x,y) for x,y in cnt_cluster])
        # medial_axis = morph.medial_axis(cluster)
        # =========== Not used (below)
        #sk = morph.skeletonize(cluster)
        # ===============================================
        # Find the corners with the corner_fast method
        success = False
        while not success:
            im_corners, success = self._find_corners(cluster, n_fast, threshold_fast)
            n_fast -= 1
            if not success and not isinstance(im_corners, np.ndarray):
                return [], [], None
        corners_index = []
        for i in range(2):
            # Find the contour
            cnt_corner = measure.find_contours(im_corners == i+1, 0.5)[0]
            # Create a string for the coordinates
            cnt_corner_2_string = np.array(["%s,%s" % (x,y) for x,y in cnt_corner])
            # Find where the corner matches the contour
            # is_matching_points = np.in1d(cnt_corner_2_string, cnt_cluster_2_string)
            # matching_points = cnt_corner_2_string[is_matching_points]
            is_matching_points = np.in1d(cnt_cluster_2_string, cnt_corner_2_string)
            matching_points = cnt_cluster_2_string[is_matching_points]
            if len(matching_points) == 0:
                return [], [], None
            index_matching_point = len(matching_points) // 2
            # Get the string of the cornet and find it in the cluster contour
            corner_string = matching_points[index_matching_point]
            corner_index = np.argwhere(cnt_cluster_2_string==corner_string)[0][0]
            corners_index.append(corner_index)
        # 5. Split the cluster contour in two sub-arrays
        i0, i1 = np.sort(corners_index)
        cnt_cluster_rolled = np.roll(cnt_cluster, len(cnt_cluster)-i0, axis=0)
        l0, l1 = cnt_cluster_rolled[:i1-i0+1], cnt_cluster_rolled[i1-i0:]
        l0_distance = self._mean_distance(l0, self.ref_point)
        l1_distance = self._mean_distance(l1, self.ref_point)
        if self.motion == 'downward' and (l0_distance > l1_distance):
                l0, l1 = l1, l0
        if test:
            fig, ax = plt.subplots(1,1)
            ax.imshow(cluster, 'gray')
            ax.plot(l0[:,1],l0[:,0],'oy', label='start')
            ax.plot(l1[:,1],l1[:,0],'ob', label='end')
            axs = 0.9*np.min(l1[:,1]), 1.1*np.max(l1[:,1]), 1.1*np.max(l1[:,0]), 0.9*np.min(l1[:,0])
            ax.axis(axs)
            ax.legend()
        return l0, l1, im_corners

    def _largest_cluster(self, im):
        """
        find the largest cluster in a image
        """
        im, n_clusters = mahotas.label(im)
        if n_clusters == 1:
            return im, np.sum(im)
        else:
            sizes = mahotas.labeled.labeled_size(im)[1:]
            i = np.argmax(sizes)
            return im==i+1, sizes[i]

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

    def _zeros(self, threshold, method='full_histogram'):
        """
        Find the zeros of the histogram
        i.e. where the signal == threshold
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

    def plotEventsAndClusters(self, cluster_threshold=5, method='full_histogram', 
                                fig=None, axs=None, title=None, with_cluster_number=True):
        """
        method: str
            sub_cluster: detect if there is a sub_cluster larger than the threshold
            full_histogram: detect if there total number of switches is larger than the threshold
        """
        if not self.is_histogram:   
            self.plotHistogram(self._switchTimes2D)
        x0s, x1s = self._zeros(cluster_threshold, method=method)
        self.events_and_clusters = mke.EventsAndClusters(self._switchTimes2D)
        self.events_and_clusters.get_events_and_clusters(min_cluster_size=0, cluster_limits=zip(x0s,x1s))
        self.events_and_clusters.plot_maps(self._colorMap, zoom_in_data=self.zoom_in_data, 
                                            fig=fig, axs=axs, title=title, with_cluster_number=True)


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

