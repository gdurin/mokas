import pickle
import numpy as np
import mahotas
import getLogDistributions as gLD
import matplotlib.pyplot as plt
import itertools
import scipy.spatial as spatial
from scipy.spatial.distance import cdist
from skimage.morphology import remove_small_holes
from skimage.draw import line_aa


NNstructure = np.ones((3,3))

filename = "switch2D_05.pkl"

with open(filename, 'rb') as f:
    switch2D = pickle.load(f)

sw = np.unique(switch2D)[1:]
events_sizes = np.array([])

plt.figure()
q = switch2D >= sw[0]
im, n_cluster = mahotas.label(q, NNstructure)
n = n_cluster + 1
plt.imshow(im, interpolation='None')

rows, cols = im.shape
distance0 = (rows**2 + cols**2)

for i, j in itertools.combinations(range(1, n), 2):
    distance = distance0
    print(i,j)
    im_i = im == i
    im_j = im == j
    p_i = mahotas.labeled.bwperim(im_i)
    p_j = mahotas.labeled.bwperim(im_j)
    indices_i = np.asarray(np.nonzero(p_i)).T
    indices_j = np.asarray(np.nonzero(p_j)).T
    mytree = spatial.cKDTree(indices_i)
    dist, indexes = mytree.query(indices_j)
    imin = np.argmin(dist)
    q_i = mytree.data[indexes[imin]]
    mytree = spatial.cKDTree(indices_j)
    dist, indexes = mytree.query(indices_i)
    imin = np.argmin(dist)
    q_j = mytree.data[indexes[imin]]
    print(q_i, q_j)
    x_i, y_i = q_i.astype(int)
    x_j, y_j = q_j.astype(int)
    rr, cc, val = line_aa(x_i,y_i,x_j,y_j)#q_i[0],q_i[1],q_j[0],q_j[1]
    im[rr,cc] = 1
    # im[x_i:x_j:np.sign(x_j-x_i),y_i] = 1
    # im[x_j,y_i:y_j:np.sign(y_j-y_i)] = 1




# im[ im>0 ] = 1
# im[ im==0 ] = 2
# im, n_cluster = mahotas.label(im, NNstructure)
q = im > 0
q = remove_small_holes(q)
im, n_cluster = mahotas.label(~q, NNstructure)
im = mahotas.labeled.remove_bordering(im)
plt.imshow(im,interpolation='None')
# # calculating the distance matrix
# distance_matrix = np.zeros((n-1, n-1), dtype=np.float)   
# for i, j in itertools.combinations(range(n-1), 2):
#     # use squared Euclidean distance (more efficient), and take the square root only of the single element we are interested in.
#     d2 = cdist(indexes[i], indexes[j], metric='sqeuclidean') 
#     distance_matrix[i, j] = distance_matrix[j, i] = d2.min()**0.5

# # mapping the distance matrix to labeled IDs (could be improved/extended)
# labels_i, labels_j = np.meshgrid( range(1, n), range(1, n))  
# results = np.dstack((labels_i, labels_j, distance_matrix)).reshape((-1, 3))

# print(distance_matrix)
# print(results)

print("%i clusters" % n_cluster)
plt.show()

