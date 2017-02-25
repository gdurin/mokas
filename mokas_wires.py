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
        self.imParameters['rotation'] = float(self.default['rotation'])
        if n_wire > self.n_wires:
            print("Number of the wire not available (1-%i)" % self.n_wires)
        nwire = "n%i" % n_wire
        nw = self.config[nwire]
        crop_upper_left_pixel = tuple([int(n) for n in nw['crop_upper_left_pixel'].split(",")])
        crop_lower_right_pixel = tuple([int(n) for n in nw['crop_lower_right_pixel'].split(",")])
        self.imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        self.experiments = [int(n) for n in nw['experiments'].split(",")]
        analysis = self.config['Analysis']
        self.motion = analysis['motion']
        self.threshold = float(analysis['threshold'])


class Wires(StackImages):
    """
    define a proper class to handle
    the sequence of images
    taken from wires
    """
    def __init__(self, motion='downward', **imParameters):
        StackImages.__init__(self, **imParameters)
        self.motion = motion
        n, rows, cols = self.shape
        if motion == 'downward':
            self.ref_point = (0, cols//2)
        elif motion == 'upward':
            self.ref_point = (rows, cols//2)
        elif motion == 'leftward':
            self.ref_point = (rows//2, cols)
        elif motion == 'rightward':
            self.ref_point = (rows//2, 0)


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
            if not success and im_corners == False:
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

    def find_contours(self, lines_color=None, invert_y_axis=True, step_image=1,
                        consider_events_around_a_central_domain=True, 
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
        self.contours = {}
        switch0 = self.switches[0]
        cluster = self._switchTimes2D == switch0
        for switch in self.switches:
            cluster += self._switchTimes2D == switch
            cluster, cluster_size = self._largest_cluster(cluster)
            cnts = measure.find_contours(cluster, 0.5)[0]
            self.contours[switch] = cnts
            X,Y = cnts[:,1], cnts[:,0]
            ax.plot(X,Y,c='k',antialiased=True,lw=1)
        self.is_find_contours = True
        if invert_y_axis:
            ax.invert_yaxis()
        plt.show()