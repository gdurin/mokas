import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import pandas as pd

data = {}
curr_nij = [("0.137A", "044"),
            ("0.146A", "033"),
            ("0.157A", "022"),
            ("0.165A", "022"),
            ("0.165A", "015"),
            ]
fields = {"0.137A": "0.13", "0.146A": "0.14", "0.157A": "0.15", "0.165A": "0.16"}
curr_nij = curr_nij[:4]
n_set = "Set1"
PS_types = ["events", "nij", "nij_filtered", "touch"]
PS_type = PS_types[2]

hname = '/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/Results_NonIrr_Feb2018.hdf5'

store = pd.HDFStore(hname)
ac = {"0.137A": .85, "0.146A": .9, "0.157A": 1, "0.165A": 1}
clrs = ['b', 'orange', 'g', 'r']

fig, ax = plt.subplots(1,1, figsize=(8.7,5.8))
for i,q in enumerate(curr_nij):
    current, n_ij = q
    if "nij" in PS_type:
        group = "%s/%s/PS_%s_%s" % (current, n_set, PS_type, n_ij)
        _n_ij = "0.%s" % n_ij[1:]
        lb = "Field: %s mT, n_ij: %s" % (fields[current], _n_ij)
    else:
        group = "%s/%s/PS_%s" % (current, n_set, PS_type)
        lb = "Field: %s mT" % fields[current]
    q = store.get(group)
    data[current] = q
    S, PS, PS_err = q.S, q.PS, q.PS_err
    if PS_type == 'nij_filtered':
        A = ac[current]
    else:
        A = 1.
    ax.loglog(S, A*PS, 'o', label=lb, c=clrs[i])
    #ax.errorbar(S, PS, PS_err, fmt="none")
    #    S_calc = np.logspace(np.log10(np.min(S)), np.log10(np.max(S)), 2*len(S))
    #    ax.loglog(S_calc, sd.y(params, S_calc), '--', color=color)
    ax.legend(loc=3)
xlabel, ylabel = "$S_{Clust}$", "$P(S_{Clust})$"
if PS_type == 'events':
    xlabel, ylabel = 'event size (S)', "$P(S)$"
    ax.plot(S, .015*(S/10.)**(-1.17) * np.exp(-S/37.), 'k--')
elif PS_type == 'nij_filtered':
    A = 0.18
    imax = 6
    lw = 1.5
    ax.plot(S[:-imax], A*(S[:-imax]/10.)**(-1.287) * np.exp(-S[:-imax]/170.), 'b--', lw=lw)
    imax = 3
    ax.plot(S[:-imax], A*(S[:-imax]/10.)**(-1.287) * np.exp(-S[:-imax]/300.), '--', lw=lw, c='orange')
    ax.plot(S, (A+.01)*(S/10.)**(-1.287) * np.exp(-(S/300.)**0.6), 'g--', lw=lw)
    ax.plot(S, (A+.01)*(S/10.)**(-1.287) * np.exp(-(S/380.)**0.6), 'r--', lw=lw)
ax.axis((7,4000,1e-6,0.3))
ax.set_xlabel(xlabel, size=20)
ax.set_ylabel(ylabel, size=20)
if "nij" in PS_type:
    ax.set_title("Size distribution of clusters")
else:   
    ax.set_title("Size distribution of %s" % PS_type)

ax.grid(True)
plt.show()

store.close()