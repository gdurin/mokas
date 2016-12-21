import pickle
import numpy as np
import mahotas
import getLogDistributions as gLD
import matplotlib.pyplot as plt

NNstructure = np.ones((3,3))

filename = "switch2D_05.pkl"

with open(filename, 'rb') as f:
    switch2D = pickle.load(f)

sw = np.unique(switch2D)[1:]
events_sizes = np.array([])

for switch in sw:
    q = switch2D == switch
    im, n_cluster = mahotas.label(q, NNstructure)
    sizes = mahotas.labeled.labeled_size(im)[1:]
    events_sizes = np.concatenate((events_sizes, sizes))

plt.figure()
q = switch2D >= sw[0]
im, n_cluster = mahotas.label(q, NNstructure)
print("%i clusters" % n_cluster)
plt.imshow(q)
print("We have collected %i events" % len(events_sizes))

# Calculate and plot the distributions of clusters and avalanches
D_x, D_y, D_yerr = gLD.logDistribution(events_sizes, log_step=0.1, 
                               first_point=1., normed=True)
# Plots of the distributions
plt.figure()
plt.loglog(D_x,D_y,'o',label='events')
plt.errorbar(D_x,D_y,D_yerr)
plt.loglog(D_x,0.14*D_x**-1.17*np.exp(-D_x/50),'-', label=r'S^{-1.17}')
plt.legend()
plt.grid()
plt.show()

