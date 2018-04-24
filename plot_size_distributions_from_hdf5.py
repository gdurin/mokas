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
PS_type = PS_types[0]

hname = '/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/Results_NonIrr_Feb2018.hdf5'

store = pd.HDFStore(hname)

fig, ax = plt.subplots(1,1)
for current, n_ij in curr_nij:
    if "nij" in PS_type:
        group = "%s/%s/PS_%s_%s" % (current, n_set, PS_type, n_ij)
        lb = "Field: %s mT, n_ij: %s" % (fields[current], n_ij)
    else:
        group = "%s/%s/PS_%s" % (current, n_set, PS_type)
        lb = "Field: %s mT" % fields[current]
    q = store.get(group)
    data[current] = q
    S, PS, PS_err = q.S, q.PS, q.PS_err

    ax.loglog(S, PS, 'o', label=lb)
    #ax.errorbar(S, PS, PS_err, fmt="none")
    #    S_calc = np.logspace(np.log10(np.min(S)), np.log10(np.max(S)), 2*len(S))
    #    ax.loglog(S_calc, sd.y(params, S_calc), '--', color=color)
    ax.legend(loc=3)
if PS_type == 'events':
    xlabel, ylabel = 'event size (S)', "$P(S)$"
    ax.plot(S, .015*(S/10.)**(-1.17) * np.exp(-S/37.), 'k--')
else:
    xlabel, ylabel = "$S_{Clust}$", "$P(S_{Clust})$"
ax.set_xlabel(xlabel, size=20)
ax.set_ylabel(ylabel, size=20)
ax.set_title("Size distribution of %s" % PS_type)

ax.grid(True)
plt.show()

store.close()