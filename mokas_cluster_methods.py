import os
import numpy as np
import mahotas
import skimage.morphology as morph
import skimage.feature as feature
from skimage import measure
import matplotlib.pyplot as plt

def get_lenght_and_curvature(line):
    """
    Meausure the length of a contour line
    in pixel units
    Get the curvature is required
    """
    try:
        x, y = line[:,1], line[:,0]
        lenght = np.sum(((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)**0.5)
        curvature, b, c = np.polyfit(x, y, 2)
    except TypeError:
        return None, None
    return lenght, -curvature
    

def largest_cluster(im, NNstructure=None):
    """
    find the largest cluster in a image
    """
    if NNstructure is None:
        NNstructure = np.ones((3,3))
    im, n_clusters = mahotas.label(im, NNstructure)
    if n_clusters == 1:
        return im, np.sum(im)
    else:
        sizes = mahotas.labeled.labeled_size(im)[1:]
        i = np.argmax(sizes)
        return im==i+1, sizes[i]

def mean_distance(cnt, ref_point):
    """
    Find the mean distance between a contour
    and a reference point
    """
    y,x = np.hsplit(cnt,2)
    y0, x0 = ref_point
    mean_distance = np.mean(((y-y0)**2 + (x-x0)**2)**0.5)
    return mean_distance

def find_corners(cluster, n_fast=12, threshold_fast=0.1, method='farthest'):
    """
    find 2 corners in the cluster
    Method:
    'farthest' from the center of mass
    'largest' between all the corners
    """

    # Find the corners with the corner_fast method
    # Be sure there are no holes inside
    cluster = morph.remove_small_holes(cluster)
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

def get_upper_and_lower_contour(cluster, motion, ref_point, n_fast=12, threshold_fast=0.1, 
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
        cluster, cluster_size = largest_cluster(cluster)
    cluster = morph.remove_small_holes(cluster)
    cnt_cluster = measure.find_contours(cluster,0.5)[0]
    cnt_cluster_2_string = np.array(["%s,%s" % (x,y) for x,y in cnt_cluster])
    # ===============================================
    # Find the corners with the corner_fast method
    success = False
    while not success:
        im_corners, success = find_corners(cluster, n_fast, threshold_fast)
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
    l0_distance = mean_distance(l0, ref_point)
    l1_distance = mean_distance(l1, ref_point)
    if motion == 'downward' and (l0_distance > l1_distance):
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
