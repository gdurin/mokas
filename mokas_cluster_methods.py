import os
import numpy as np
import mahotas
import skimage.morphology as morph
import skimage.feature as feature
from skimage import measure
import matplotlib.pyplot as plt
import heapq



k0 = np.array([0,0,1,1,1,0,0])
k1 = np.array([0,1,0,0,0,1,0])
kc =  np.array([1,0,0,0,0,0,1])
kernel = np.vstack((k0,k1,kc,kc,kc,k1,k0))

#n_clusters_as_type = {'0000': 2, '0100':1, }

def get_length(line):
    x, y = line[:,1], line[:,0]
    length = np.sum(((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)**0.5)
    return length

def get_length_and_curvature(line):
    """
    Meausure the length of a contour line
    in pixel units
    Get the curvature is required
    """
    try:
        x, y = line[:,1], line[:,0]
        length = get_length(line)
        curvature, b, c = np.polyfit(x, y, 2)
    except TypeError:
        return None, None
    return length, -curvature
    

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



def _find_corners(cluster, cluster_type, n_fast=12, threshold_fast=0.1, method='suppression'):
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
    try:
        cf = feature.corner_fast(cluster, n_fast, threshold_fast)
        # Now find the clusters associated to them
        im, n_clusters = mahotas.label(cf)
    except OverflowError:
        print("There is an Overflow Error")
        return False, False
    # Assestment of n. of corners based on the cluster_type
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
        elif method == 'suppression':
            pass
        else:
            print("Not implemented")
            return None, False
        im, n_left = mahotas.labeled.relabel(im)
        try:
            assert n_left == 2
        except AssertionError:
            return im, False
    return im, True

def _non_maximal_suppression(im_corners, cluster, NN=3):
    """
    Non-maximal Suppression
    Detecting multiple interest points in adjacent locations
    is another problem. It is solved by using Non-maximum Suppression.

    Compute a score function, V for all the detected feature points. 
    V is the sum of absolute difference between p and 16 surrounding pixels values.
    Consider two adjacent keypoints and compute their V values.
    Discard the one with lower V value.
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html

    In our case, we select the one with the highest V value

    Output:
        Coordinate (row, col) of the corner
    """
    cluster = cluster.astype(np.int)
    corners_rows, corners_cols = np.nonzero(im_corners)
    V = []
    for row, col in zip(corners_rows, corners_cols):
        cl = cluster[row-NN:row+NN+1, col-NN:col+NN+1]
        V.append(np.sum(np.abs(1-cl*kernel)))
    V = np.array(V)
    i = V.argmax()
    print("Average V: %f, %f" % (np.mean(V), np.median(V)))
    return corners_rows[i], corners_cols[i]

def get_corner_index(im_corners, cnt_cluster):
    cnt_corner = measure.find_contours(im_corners, 0.5, fully_connected='low')[0]
    # Create a string for the coordinates
    cnt_cluster_2_string = np.array(["%s,%s" % (x,y) for x,y in cnt_cluster])
    cnt_corner_2_string = np.array(["%s,%s" % (x,y) for x,y in cnt_corner])
    # Find where the corner matches the contour
    is_matching_points = np.in1d(cnt_cluster_2_string, cnt_corner_2_string)
    if sum(is_matching_points) == 0:
        return None
    matching_points = cnt_cluster_2_string[is_matching_points]
    index_matching_point = len(matching_points) // 2
    # Get the string of the corner and find it in the cluster contour
    corner_string = matching_points[index_matching_point]
    corner_index = np.argwhere(cnt_cluster_2_string==corner_string)[0][0]
    return corner_index

def get_contours(cluster, cluster_type, threshold=0.5, connection='low'):
    cnts_cluster = measure.find_contours(cluster, threshold, fully_connected=connection)
    if cluster_type != '1100':
        if len(cnts_cluster) > 1:
            # If there are several contours, keep the longest
            lens = np.array([len(cnt) for cnt in cnts_cluster])
            i0 = np.argmax(lens)
            # We need to 
            return cnts_cluster[i0], cluster
        else:
            #print(len(cnts_cluster))
            return cnts_cluster[0], cluster
    else:
        return cnts_cluster, cluster


def get_upper_and_lower_contour(cluster, cluster_type, ref_point=None, motion='downward',
                                n_fast=12, threshold_fast=0.1, test=False):
    """
    This is really a tough problem!
    Find the initial and final DW position in a cluster

    Method:
    1. Find the corners (to be ckeck in limit cases)
    2. Find the contour of the corners and of the cluster
    3. Find the common elements of the two contous
    4. Decide the position of the corner (the middle element)
    5. Split the cluster contour in two sub-arrays
        
    Extract FAST corners for a given image.
    Parameters: 
    ===========
    cluster : 2D ndarray
        the image to analyse
    motion : str
        Motion of the cluster. Assumed to be 'downard'
        To be checked for other cases
    ref_point : tuple
        Reference to calculate the distance of the two counturs
    cluster_type : str
        String to know if the cluster touches the 
    n_fast : int
        Minimum number of consecutive pixels out of 16 pixels
        on the circle that should all be either brighter or darker w.r.t testpixel.
        A point c on the circle is darker w.r.t test pixel p if Ic < Ip - threshold 
        and brighter if Ic > Ip + threshold. 
        Also stands for the n in FAST-n corner detector.
    threshold_fast : float
        Threshold used in deciding whether the pixels on the circle are brighter, 
        darker or similar w.r.t. the test pixel. Decrease the threshold when more corners
        are desired and vice-versa.
    test : bool
        Plot the contours to test
    """

    failure = [np.NaN, np.NaN, np.NaN, False]
    im, n_clusters = mahotas.label(cluster, np.ones((3,3)))
    if n_clusters != 1:
        raise ValueError("The cluster is made of sub_clusters, which is not permitted here")
    if ref_point is None:
        rows, cols = cluster.shape
        ref_point = 0, cols/2
    cluster = morph.remove_small_holes(cluster)
    # Find the contour
    #cnts_cluster = measure.find_contours(cluster,0.5, fully_connected='high')
    cnt_cluster, cluster = get_contours(cluster, cluster_type, connection='low')
    if cluster_type == '1100':
        l0, l1 = cnt_cluster[0], cnt_cluster[1]
    else:
        # It is better to close the contour for the 01 and 10 clusters
        if cluster_type[1] == '1':
            cnt_cluster = np.vstack((cnt_cluster, cnt_cluster[0]))
        elif cluster_type[0] == '1':
            cnt_cluster = np.vstack((cnt_cluster[-1], cnt_cluster))
        # This is a fantastic method to get the cluster pixels inside the contour only!
        cluster = measure.grid_points_in_poly(cluster.shape, cnt_cluster)
        # Find the corners
        cf = feature.corner_fast(cluster, n_fast, threshold_fast)
        if np.sum(cf) == 0:
            cf = feature.corner_fast(cluster, n_fast-1, threshold_fast)
        im_cf, n_clusters_cf = mahotas.label(cf, np.ones((3,3)))
        if not n_clusters_cf:
            failure[-1] = "No corners found"
            print("No corners found")
            print(n_clusters_cf, cluster_type, np.sum(im_cf))
            return failure
        ##########################################################
        if cluster_type == '0000':
            i_corners = []
            if n_clusters_cf == 1:
                #print("Only 1 corner")
                failure[-1] = 'Only 1 corner'
                return failure
            elif n_clusters_cf == 2:
                i_range = range(1,3)
            elif n_clusters_cf > 2:
                # Take the two im_cf at the rightmost and leftmost positions
                h_pos = [np.nonzero(im_cf==i)[1] for i in range(1,n_clusters_cf+1)]
                leftmost = np.array([np.min(h) for h in h_pos]).argmin() + 1
                rightmost = np.array([np.max(h) for h in h_pos]).argmax() + 1
                i_range = [leftmost, rightmost]
            for i in i_range:
                corner_index = get_corner_index(im_cf==i, cnt_cluster)
                if corner_index is None:
                    failure[-1] = "No corner's index"
                    return failure
                i_corners.append(corner_index)                    
            # 5. Split the cluster contour in two sub-arrays
            i0, i1 = np.sort(i_corners)
            cnt_cluster_rolled = np.roll(cnt_cluster, len(cnt_cluster)-i0, axis=0)
            l0, l1 = cnt_cluster_rolled[:i1-i0+1], cnt_cluster_rolled[i1-i0:]
        elif '1' in cluster_type[:2]:
            if n_clusters_cf > 1 and cluster_type == '1000':
                h_pos = [np.nonzero(im_cf==i)[1] for i in range(1,n_clusters_cf+1)]                
                rightmost = np.array([np.max(h) for h in h_pos]).argmax() + 1
                im_cf = im_cf == rightmost
            elif n_clusters_cf > 1 and cluster_type == '0100':
                h_pos = [np.nonzero(im_cf==i)[1] for i in range(1,n_clusters_cf+1)]
                leftmost = np.array([np.min(h) for h in h_pos]).argmin() + 1
                im_cf = im_cf == leftmost
            i0 = get_corner_index(im_cf, cnt_cluster)
            if i0 is None:
                failure[-1] = "No corner found"
                return failure
            l0, l1 = cnt_cluster[:i0+1], cnt_cluster[i0:]
    
    ####################################
    try:
        l0_distance = mean_distance(l0, ref_point)
        l1_distance = mean_distance(l1, ref_point)
    except:
        print("l0 not found for type %s" % cluster_type)
    if motion == 'downward' and (l0_distance > l1_distance):
        l0, l1 = l1, l0
    l_linear = mean_distance(l0[0],l0[-1]) #Isn't nice?   
    success = True
    if test and '11' == cluster_type[:2]:
        fig, ax = plt.subplots(1,1)
        ax.imshow(cluster, 'gray', interpolation='nearest')
        ax.plot(l0[:,1],l0[:,0],'oy', label='start')
        ax.plot(l1[:,1],l1[:,0],'ob', label='end')
        axs = 0.9*np.min(l1[:,1]), 1.1*np.max(l1[:,1]), 1.1*np.max(l1[:,0]), 0.9*np.min(l1[:,0])
        ax.axis(axs)
        ax.legend()
        plt.show()
    return l0, l1, l_linear, success

