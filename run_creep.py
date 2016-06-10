from __future__ import print_function
import os,sys
import configparser
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as mpl_colors
import pandas as pd
from visualBarkh import StackImages
import polar
import iniConnector as iniC


def get_cmap(N):
    """Returns a function that maps each index 
    in 0, 1, ... N-1 to a distinct RGB color.
    http://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    """
    color_norm  = mpl_colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    pColor = [scalar_map.to_rgba(i)[:3] for i in range(N)]
    map_index_to_rgb_color = [[int(col[0]*255),int(col[1]*255),int(col[2]*255)] for col in pColor]
    return map_index_to_rgb_color

def get_colors(num_colors, palette='hue',norm=False):
	black = np.array(3*[0])
	white = np.array(3*[255])
	if palette == 'hue':
	    colors=[]
	    for i in np.arange(0., 360., 360. / num_colors):
	        hue = i/360.
	        lightness = (50 + np.random.rand() * 10)/100.
	        saturation = (90 + np.random.rand() * 10)/100.
	        col = colorsys.hls_to_rgb(hue, lightness, saturation)
	        c = [int(col[0]*255),int(col[1]*255),int(col[2]*255)]
	        colors.append(c)
	    colors = np.random.permutation(colors)
	elif palette == 'pastel':
		colors = (np.random.randint(0, 256, (num_colors,3)) + white) / 2
		#colors = get_cmap(num_colors)
	colors = np.vstack((black,colors))
	if norm:
		colors = colors/255.
	return colors

def get_rowcols(n):
	if n <= 11:
		return 2, np.int(np.ceil(n/2.))
	elif n <=17:
		return 3, np.int(np.ceil(n/3.))
	elif n <=23:
		return 4, np.int(np.ceil(n/4.))
	elif n >=23:
		return 5, np.int(np.ceil(n/5.))

class Creep:
	def __init__(self, iniFilepath, Bz, gray_threshold=None):
		print("Reading ini file....",end="");
		self.rootDir = os.path.dirname(iniFilepath)
		self.measData = iniC.connect_to(iniFilepath,Bz)
		self.imParameters = self.measData.imageParameters
		self.Bz_mT = self.measData.Bz_mT
		self.full_title = self.measData.material_full + " - " + "$B_z = %i mT$" % self.Bz_mT
		self.Bx_unit = self.measData.Bx_unit
		self.step_in_frames = self.measData.step_in_frames
		self.microns_per_pixel = self.measData.microns_per_pixel
		self.frame_rate = self.measData.frame_rate
		if gray_threshold is None:
			try:
				self.gray_threshold = self.measData.gray_threshold
			except:
				print("Gray level not provided")
				self.gray_threshold = float(raw_input("gray_threshold?"))
		# Select the InP fields
		self.Bx_s = self.measData.Bx_s
		isProblem = self.check_dirs()
		if isProblem:
			print()
			print("Exiting....")
			sys.exit()
		else:
			# Finish initialization
			self.all_contours = {}
			self.all_centers = {}
			print("Done.")
	
	def check_dirs(self):	
		# Need to check now that all the files exist:
		isProblem = False
		print("Check directories",end="")
		for Bx in self.Bx_s:
			subDir = self.measData.varsBx[Bx]['subDir']
			if not os.path.isdir(os.path.join(self.rootDir,subDir)):
				print("{0} not found".format(subDir))
				isProblem = True
			else:
				print(".",end="")
	
	def get_color_sequence(self):
		colormap = plt.cm.gist_ncar
		colors = [colormap(i) for i in np.linspace(0, 0.9, len(self.Bx_s))]
		return colors
	
	def plot_results(self):
		"""
		Plot the different images for creep calculation
		"""
		# Prepare to plot
		n_alphas = 36 + 1
		plt.close("all")
		self.figs = []
		self.velocities_mean_error = {}
		print("Preparing plots",end="")
		rows, cols = get_rowcols(len(self.Bx_s))
		self.fig1, self.axs1 = plt.subplots(rows,cols,sharex=True, sharey=True) # ColorImages
		self.figs.append(self.fig1)
		print(".",end="")
		self.fig2, self.axs2 = plt.subplots(rows,cols) # Histograms
		self.figs.append(self.fig2)
		print(".",end="")
		self.fig3, self.axs3 = plt.subplots(rows,cols,sharex=True, sharey=True) # Contours
		self.figs.append(self.fig3)
		self.fig3b, self.axs3b = plt.subplots(rows,cols,sharex=True, sharey=True) # Contours
		self.figs.append(self.fig3b)
		print(".",end="")
		self.fig4, self.axs4 = plt.subplots(rows,cols,sharex=True, sharey=True) # Displacements (absolute)
		self.figs.append(self.fig4)
		print(".",end="")
		self.fig5, self.axs5 = plt.subplots(rows,cols,sharex=True, sharey=True) # Displacements/velocity (relative)
		self.figs.append(self.fig5)
		print(".",end="")
		self.fig6, self.axs6 = plt.subplots(1,2) # Velocity
		self.figs.append(self.fig6)
		print(".",end="")
		self.fig7, self.axs7 = plt.subplots(rows,cols,sharex=True, sharey=True) # velocity (relative)
		self.figs.append(self.fig7)
		print("Done")

		colors = self.get_color_sequence()

		self.imArray_collector = {}
		for n,Bx in enumerate(self.Bx_s):
			print(20*"*")
			print("Bx = %i %s, Bz = %i mT" % (Bx,self.Bx_unit,self.Bz_mT))
			self.imParameters['firstIm'] = self.measData.varsBx[Bx]['firstIm']
			self.imParameters['lastIm'] = self.measData.varsBx[Bx]['lastIm']
			pulse_duration = self.measData.varsBx[Bx]['pulse_duration']
			subDir = self.measData.varsBx[Bx]['subDir']
			self.imParameters['subDirs'] = [self.rootDir, subDir, "", "", ""]

			title = "Bx = %i %s, p = %s s" % (Bx,self.Bx_unit,pulse_duration)
			
			imArray = StackImages(**self.imParameters)
			self.imArray_collector[Bx] = imArray
			imArray.width='all'
			imArray.useKernel = 'step'
			imArray.kernelSign = -1

			# Plot the subplots
			if n==0:
				nImages = ((self.imParameters['lastIm'] - self.imParameters['firstIm'])*5)
				pColor = get_colors(nImages,'pastel',norm=True)


			i, j = np.int(np.floor(n/cols)), n%cols
			# Figure 1 : color Image of DW motion
			imArray.showColorImage(self.gray_threshold,palette=pColor,
				plotHist=None,plot_contours=False,fig=self.fig1,ax=self.axs1[i,j],title=title,noSwitchColor='black')
			
			# Figure 2 : plot the histogram
			imArray.plotHistogram(imArray._switchTimesOverThreshold,fig=self.fig2,ax=self.axs2[i,j],title=title,ylabel=None)
			
			# Calculate the contours
			imArray.find_contours(lines_color='k',remove_bordering=True,
				plot_centers_of_mass=True, fig=self.fig3,ax=self.axs3[i,j],title=title)
			imArray.find_contours(remove_bordering=True,reference='center_of_mass',rescale_area=True,
				fig=self.fig3b,ax=self.axs3b[i,j],title=title)

			self.all_contours[Bx] = imArray.contours
			self.all_centers[Bx] = imArray.centers_of_mass
			center = imArray.centers_of_mass[0] # Center of mass of the initial domain 

			#Plot the central domain
			cnts0 = imArray.contours[0]
			self.axs1[rows-1,cols-1].plot(cnts0[:,1],cnts0[:,0],c=colors[n],lw=1)
			self.axs3[rows-1,cols-1].plot(cnts0[:,1],cnts0[:,0],c=colors[n],lw=.5)

			#Plot the external contour
			lastKey = sorted(imArray.contours.keys())[-1]
			cnts = imArray.contours[lastKey]
			xcnts, ycnts = cnts[:,1], cnts[:,0]
			Xmin, Xmax, Ymin, Ymax = min(xcnts), max(xcnts), min(ycnts), max(ycnts)
			if not n:
				xmin, xmax, ymin, ymax = Xmin, Xmax, Ymin, Ymax
			if Xmax > xmax: xmax = Xmax
			if Ymax > ymax: ymax = Ymax
			if Xmin < xmin: xmin = Xmin
			if Ymin < ymin: ymin = Ymin
			if int(Bx) == Bx:
				label = int(Bx)
			else:
				label = Bx
			for ax in [self.axs1,self.axs3]:
				ax[rows-1,cols-1].plot(xcnts,ycnts,c=colors[n],lw=1,label=label)

			# Plot dispacements in polar coordinates from the center
			theta, r, frames = polar.plot_displacement(imArray.contours,origin=center,reference='center',
				swope_xy=True,fig=self.fig4,ax=self.axs4[i,j],title=title,step_in_frames=self.step_in_frames)
			
			# plot last contours
			self.axs4[rows-1,cols-1].plot(theta/np.pi*180,r,c=colors[n],lw=2,label=label)
			self.axs4[i,j].plot(theta/np.pi*180,r,c=colors[n],lw=2)
			
			# Plot dispacements in polar coordinates from the nucleated domain
			theta, last_r, frames = polar.plot_displacement(imArray.contours,origin=center,reference='nucleated_domain',
				swope_xy=True,fig=self.fig5,ax=self.axs5[i,j],title=title,step_in_frames=self.step_in_frames)
			self.axs5[rows-1,cols-1].plot(theta/np.pi*180,last_r,c=colors[n],lw=2,label=label)
			self.axs5[i,j].plot(theta/np.pi*180,last_r,c=colors[n],lw=2)

			# Plot the mean velocity as a function of theta
			v = polar.calc_velocity(imArray.contours,origin=center,n_new_thetas=n_alphas,swope_xy=True)
			v_mean, v_error = polar.calc_mean_error_velocity(v)
			thetas = v.columns
			if n == 0:
				self.velocities = pd.DataFrame(v_mean,columns=[Bx])
				self.velocities_error = pd.DataFrame(v_error,columns=[Bx])
			else:
				self.velocities[Bx] = v_mean
				self.velocities_error[Bx] = v_error
			polar.plot_mean_velocity(thetas,v_mean,v_error,fig=self.fig7,ax=self.axs7[i,j],title=title,color=colors[n])
			polar.plot_mean_velocity(thetas,v_mean,v_error,fig=self.fig7,ax=self.axs7[rows-1,cols-1],
				title="",label=label,color=colors[n])

		# Plot velocities at different angles
		v = self.velocities
		v_err = self.velocities_error
		v.index = v.index*180/np.pi
		v_err.index = v_err.index*180/np.pi
		i_shift = (n_alphas - 1) / 2
		cl = get_colors(i_shift+1,'hue',True)[1:]
		for i in range(i_shift):
			col = 0*(i!=0) + 1*(i==0)
			label = "{} deg".format(v.index[i])
			self.axs6[col].errorbar(v.columns,v.iloc[i],v_err.iloc[i],fmt='--o',c=cl[i],label=label)
			label = "{} deg".format(v.index[i+i_shift])
			self.axs6[col].errorbar(v.columns,v.iloc[i+i_shift],v_err.iloc[i+i_shift],fmt='-o',c=cl[i],label=label)
		for i in range(2):
			self.axs6[i].set_ylabel("microns/s")
			self.axs6[i].set_xlabel(r"$B_x$ ({})".format(self.Bx_unit))
			l1 = self.axs6[i].legend(fontsize=12,title="Angle (deg)",
				bbox_to_anchor=(-.05, 1), loc=1, borderaxespad=0.)
			l1.draggable(True)
			
				
		limits = np.floor(xmin/100.)*100, np.ceil(xmax/100.)*100, np.ceil(ymax/100.)*100, np.floor(ymin/100.)*100
		step_angle = np.pi/10
		title_Bx = r"$B_x (%s)$" % (self.Bx_unit)
		for axs in [self.axs1,self.axs3]:
			ax = axs[rows-1,cols-1]
			ax.plot(center[1],center[0],'ko')
			polar.plot_rays((center[1],center[0]),step_angle,ax,limits)
			ax.axis(limits)
			#ax.set_aspect('equal')
			l1 = ax.legend(fontsize=12,title=title_Bx,
				bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.)
			l1.draggable(True)

		ax = self.axs7[rows-1,cols-1]
		l1 = ax.legend(fontsize=12,title=title_Bx,numpoints=1,
		 	bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.)
		l1.draggable(True)

		for axs in [self.axs4,self.axs5]:
			ax = axs[rows-1,cols-1]
			ax.grid(True)
			ax.set_xlabel("angle (deg)")
			l1 = ax.legend(fontsize=12,title=title_Bx,ncol=1,
				bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.)
			l1.draggable(True)
		self.axs4[rows-1,cols-1].set_title("Last contour")
		self.axs4[rows-1,cols-1].set_title("Velocity")

		for fig in self.figs:
			fig.suptitle(self.full_title,fontsize=30)

		plt.show()

	def plot_trajectories_centers_of_mass(self,range_Bx=None):
		fig = plt.figure()
		ax = fig.gca()
		Bx_s = np.array(self.Bx_s)
		if range_Bx is not None:
			minBx, maxBx = range_Bx
			Bx_s = Bx_s[(Bx_s >= minBx) & (Bx_s <= maxBx)]
		for Bx in Bx_s:
			centers = self.all_centers[Bx]
			reference = centers[0]
			c = pd.DataFrame.from_dict(centers,orient='index')
			x,y = c[0]-c[0][0], c[1]-c[1][0]
			ax.plot(x,y,'-o',label=Bx)
		l1 = ax.legend()
		l1.draggable(True)


def isIniFile(filename):
	return os.path.isfile(filename)




if __name__ == "__main__":
	myChoice = ['Sim','PtCoAu50Pt50','PtCoAuPt','PtCoPt'][1]
	if myChoice == 'PtCoAu50Pt50':
		#iniFilepath = "/home/gf/Meas/Creep/PtCoAu50Pt50/PtCoAuPt_exp.ini"
		iniFilepath = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/0 degree/PtCoAuPt_exp.ini"
		# Select the OoP field
		Bz = 0.780
		#Bz = 0.975 # In Volts
		#Bxs = Bxs[:3]

	elif myChoice == 'PtCoAuPt':
		iniFilepath = "/home/gf/Meas/Creep/PtCoAu50Pt50/Rotation/0 degree/PtCoAuPt_exp.ini"
		# Select the OoP field
		Bz = 0.657

	elif myChoice == 'PtCoPt':
		#iniFilepath = "/home/gf/Meas/Creep/PtCoAu50Pt50/PtCoAuPt_exp.ini"
		iniFilepath = "/home/gf/Meas/Creep/PtCoPt/M2/PtCoPt_exp.ini"
		# Select the OoP field
		Bz = 0.657


	elif myChoice == 'Sim':
		iniFilepath = "/home/gf/Meas/Creep/Simulations/disorder_sim.ini"

		# Select the OoP field
		Bz = "-20"
		#Bz_V = 0.975 # In Volts
		#Bxs = Bxs[:3]
		gray_threshold = 80

	if not isIniFile(iniFilepath):
		print("There is a problem with the ini file {}: file not found".format(iniFilepath))
		sys.exit()

	creep_data = Creep(iniFilepath, Bz)
	creep_data.plot_results()