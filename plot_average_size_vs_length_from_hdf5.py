import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import pandas as pd


def adjust_ax(ax):
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

#####################################################################
hname = '/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/Results_NonIrr_Feb2018.hdf5'
currents = ["0.137A", "0.146A", "0.157A", "0.165A"]
fields = {"0.137A": "0.13", "0.146A": "0.14", "0.157A": "0.15", "0.165A": "0.16"}
n_set = "Set1"
d_f = "1.000"
nij_s = {"0.137A": "0.44", "0.146A": "0.33", "0.157A": "0.25", "0.165A": "0.22"}
ac = {"0.137A": .85, "0.146A": .9, "0.157A": 1, "0.165A": 1}
PS_type = "S_mean"
ms = 6
zeta = 0.633
label_size = 18

store = pd.HDFStore(hname)
clrs = ['b', 'orange', 'g', 'r']

fig, ax = plt.subplots(1,1, figsize=(7,5.8))
for i, current in enumerate(currents):
    n_ij = nij_s[current]
    group = "%s/%s/df_%s/nij_%s/%s" % (current, n_set, d_f, n_ij, PS_type)
    print(group)
    #lb = "Field: %s mT, n_ij: %s" % (fields[current], nij_s[current])
    lb = "%s mT" % fields[current]
    q = store.get(group)
    x, y = q.length, q.S_mean
    A = ac[current]
    #ax.loglog(x, A*y, 'o', label=lb, c=clrs[i], ms=ms)
    ax.loglog(x, y, 'o', label=lb, c=clrs[i], ms=ms)


l = ax.legend(loc=4, fontsize='large', title="Applied fields")
plt.setp(l.get_title(),fontsize='large')
ax.loglog(x[2:-10], 1.1*y[5]*(x[2:-10]/x[5])**(1.+zeta), 'k--', lw=2)
a = np.array([2, 2e3, 4, 4e3])
ax.axis(tuple(a))
ax.set_xlabel(r"Major axis length $L\ (px)$", size=label_size)
ax.set_ylabel(r"Average Cluster size $\langle S \rangle\ (px^2)$", size=label_size)
adjust_ax(ax)


ax2 = ax.twiny()
ax2.set_xscale("log")
a2 = a * 0.3
ax2.set_xlim(a2[:2])
ax2.set_xlabel(r"Major axis length $L\ (\mu m)$", size=label_size)
ax2.get_xaxis().set_tick_params(which='both', direction='in')
#ax2.xaxis.set_ticks_position('both')


ax3 = ax.twinx()
ax3.set_yscale("log")
a3 = a * 0.09
ax3.set_ylim(a3[2:])
ax3.set_ylabel(r"Average Cluster size $\langle S \rangle\ (\mu m^2)$", size=label_size)
ax3.get_yaxis().set_tick_params(which='both', direction='in')
#ax3.yaxis.set_ticks_position('both')

ax.annotate(r'$\sim L^{1 + \zeta_{dep}}$', xy=(10, 800), size=22)



fig1, ax1 = plt.subplots(1,1, figsize=(7,5.8))
PS_type = "S_vs_l"
group = "%s/%s/df_%s/nij_%s/%s" % (current, n_set, d_f, n_ij, PS_type)
q = store.get(group)
ax1.loglog(q.Length, q.Size, 'o', c='C0', ms=ms, label=r"$\ S$")
ax1.loglog(x, y, 'o', c=clrs[i], ms=ms, label=r"$\langle S \rangle$")
lb = r'$L^{1+\zeta_{dep}}$'
ax1.loglog(x[2:-10], y[5]*(x[2:-10]/x[5])**(1.+zeta), 'k--', label=lb, lw=2)
adjust_ax(ax1)
ax1.set_xlabel(r"Major axis length $L\ (px)$", size=label_size)
ax1.set_ylabel(r"Cluster size $S\ (px^2)$", size=label_size)
ax1.axis(tuple(a))
adjust_ax(ax1)
ax1.legend(loc=4, fontsize='x-large')

ax12 = ax1.twiny()
ax12.set_xscale("log")
ax12.set_xlim(a2[:2])
ax12.set_xlabel(r"Major axis length $L\ (\mu m)$", size=label_size)
ax12.get_xaxis().set_tick_params(which='both', direction='in')

ax13 = ax1.twinx()
ax13.set_yscale("log")
ax13.set_ylim(a3[2:])
ax13.set_ylabel(r"Cluster size $S\ (\mu m^2)$", size=label_size)
ax13.get_yaxis().set_tick_params(which='both', direction='in')


fig.tight_layout()
fig1.tight_layout()
plt.show()

store.close()