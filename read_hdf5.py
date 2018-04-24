#This code:  
#	1.Saves the waiting time matrix for all fields and measurements from hdf5 file in a nested dictionary
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy.ma as ma

irradiation = "NonIrr"

Dir="/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/%s/" %(irradiation)
os.chdir(Dir)

fname="%s.hdf5" %(irradiation)
f=h5py.File(fname,'r')

list_hdf5 = []
def func(name, obj):
	if isinstance(obj, h5py.Dataset):
		if "waiting_time" in name:   #waiting_time
			list_hdf5.append(name)
f.visititems(func)	

currents=[]
measurements=[]
datasets=[]
for i in range(len(list_hdf5)):
	splitted=list_hdf5[i].split('/')
	currents.append(splitted[0])
	measurements.append(splitted[1])
	datasets.append(f[list_hdf5[i]][()])

# Wt={}
# for k in keys:
# 	Wt[k]={}
# for j in range(len(subkeys)):
# 	subkey=subkeys[j]
# 	for k, s in zip(keys,values):
# 		if k==keys[j]:
# 			Wt[k][subkey]=[s]

for i in range(len(currents)):
	print i
	array=datasets[i]
	wt_masked = np.ma.masked_where(array==0, array)
	wt_masked_flatten=wt_masked.flatten()
	wt_masked_normal=wt_masked_flatten/float(wt_masked_flatten.mean())
	bins=20
	h_wt=np.histogram(wt_masked_normal,bins=bins,normed=0)
	prob_wt=h_wt[0]/float(len(wt_masked_normal))
	bins_wt=h_wt[1][1:]  #delete first element of bins
	title=irradiation+",Current="+str(currents[i])+",Measurement No.="+str(measurements[i])
	plt.figure()
	plt.loglog(bins_wt,prob_wt,"ro",markersize=10,linewidth=3) 
	plt.title(title)
	plt.xlabel("Normalised waiting time (frames)")
	plt.ylabel("Fractions of the pixels")

	if not os.path.exists("waiting_time_figures"):
		os.makedirs("waiting_time_figures")
	os.chdir("waiting_time_figures")

	plt.savefig(irradiation+"_"+str(currents[i])+"_"+str(measurements[i])+".png")
	plt.clf()
	plt.close('all')

	#plotting the scattered plots
	wt=[]
	list_j=[]
	list_k=[]
	threshold=2*np.std(wt_masked_flatten)
	for j in range(array.shape[0]):
		for k in range(array.shape[1]):
			if array[j][k]>threshold:
				wt.append(array[j][k])
				list_j.append(j)
				list_k.append(k)
	wt=np.asarray(wt)
	array_j=np.asarray(list_j)
	array_k=np.asarray(list_k)
	array_j=array_j.max()-array_j
	area=np.pi*(0.2*wt)**2   #0 to 15 point radiuses
	plt.figure()
	plt.scatter(array_k,array_j,s=area,alpha=0.5)
	plt.title(title)
	plt.savefig("scatter_"+title+".png")
	plt.clf()
	plt.close('all')
	os.chdir('../')