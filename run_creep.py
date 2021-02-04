from __future__ import print_function
import os,sys
import configparser
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure
import bokeh.models as bkm
from bokeh.layouts import gridplot
import bokeh.palettes as palettes
import mokas_bokeh as mkb
import pandas as pd

from mokas_stackimages import StackImages
import mokas_polar as polar
import iniConnector as iniC
from mokas_colors import get_colors
from PyQt5 import QtWidgets



def get_rowcols(n):
	if n <= 11:
		return np.int(np.ceil(n/2.)), 3
	elif n <=17:
		return np.int(np.ceil(n/3.)), 4
	elif n <=23:
		return np.int(np.ceil(n/4.)), 5
	elif n >=23:
		return np.int(np.ceil(n/5.)), 5

class Creep:
	def __init__(self, Bz, iniFilepath="",  gray_threshold=None):
		if iniFilepath:
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
				print(self.imParameters)
		self.is_plot_all_figures_done = False
		
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
	
	def get_color_sequence(self, colormap=None, visualization_library='mpl'):
		n = len(self.Bx_s)
		if visualization_library == 'mpl':
			if not colormap:
				colormap = plt.cm.gist_ncar
			colors = [colormap(i) for i in np.linspace(0, 0.9, n)]
		elif visualization_library == 'bokeh':
			colors = palettes.magma(n)
			return colors
	
	def plot_results(self, colormap=None, isPlot=True, visualization_library='bokeh'):
		"""
		Plot the different images for creep calculation
		"""
		# Prepare to plot
		n_alphas = 36 + 1
		n_bokeh_plots = 7
		plt.close("all")
		self.figs = []
		self.velocities_mean_error = {}
		# Get the screen resolution
		if os.sys.platform == 'linux':
			app = QtWidgets.QApplication([])
			screen_resolution = app.primaryScreen().size()
			width, height = screen_resolution.width(), screen_resolution.height()
			dpi = app.primaryScreen().physicalDotsPerInch()
			figsize = width/2/dpi, height/2/dpi
		else:
			figsize = None
			dpi = 100
		print(width, height)
		print("Preparing plots",end="")
		rows, cols = get_rowcols(len(self.Bx_s))
		figs = {}
		if visualization_library == 'mpl':
			self.fig1, self.axs1 = plt.subplots(rows,cols,sharex=True, sharey=True, figsize=figsize, dpi=dpi, squeeze=False) # ColorImages
			self.figs.append(self.fig1)
			print(".",end="")
			self.fig2, self.axs2 = plt.subplots(rows,cols,figsize=figsize,dpi=dpi,squeeze=False) # Histograms
			self.figs.append(self.fig2)
			print(".",end="")
			self.fig3, self.axs3 = plt.subplots(rows,cols,sharex=True, sharey=False,figsize=figsize,dpi=dpi,squeeze=False) # Contours
			self.figs.append(self.fig3)
			self.fig4, self.axs4 = plt.subplots(rows,cols,sharex=True, sharey=False,figsize=figsize,dpi=dpi,squeeze=False) # Contours
			self.figs.append(self.fig4)
			print(".",end="")
			self.fig5, self.axs5 = plt.subplots(rows,cols,sharex=True, sharey=True,figsize=figsize,dpi=dpi,squeeze=False) # Displacements (absolute)
			self.figs.append(self.fig5)
			print(".",end="")
			self.fig6, self.axs6 = plt.subplots(rows,cols,sharex=True, sharey=True,figsize=figsize,dpi=dpi,squeeze=False) # Displacements/velocity (relative)
			self.figs.append(self.fig6)
			print(".",end="")
			self.fig7, self.axs7 = plt.subplots(1,2,dpi=dpi,squeeze=False) # Velocity
			self.figs.append(self.fig7)
			print(".",end="")
			#self.fig8, self.axs8 = plt.subplots(rows,cols,sharex=True, sharey=True,figsize=figsize,dpi=dpi,squeeze=False) # velocity (relative)
			#self.figs.append(self.fig8)
			out_type = 'rbg'
		elif visualization_library == 'bokeh':
			self.plots = {}
			for kk in range(1,n_bokeh_plots+1):
				self.plots[kk] = []
			out_type = 'hex'
	
		print("Done")
		# Close the figures
		for fig in self.figs:
			plt.close(fig.number)

		if not self.is_plot_all_figures_done:
				self.imArray_collector = {}

		
		

		for n,Bx in enumerate(self.Bx_s):

			pulse_duration = self.measData.varsBx[Bx]['pulse_duration']
			title = "Bx = %i %s, p = %s s" % (Bx,self.Bx_unit,pulse_duration)
		
			if self.is_plot_all_figures_done:
				imArray = self.imArray_collector[Bx]
			else:

				print(20*"*")
				print("Bx = %i %s, Bz = %i mT" % (Bx,self.Bx_unit,self.Bz_mT))
				self.imParameters['firstIm'] = self.measData.varsBx[Bx]['firstIm']
				self.imParameters['lastIm'] = self.measData.varsBx[Bx]['lastIm']
				subDir = self.measData.varsBx[Bx]['subDir']
				self.imParameters['subDirs'] = [self.rootDir, subDir, "", "", ""]
				self.imParameters['visualization_library'] = visualization_library
				imArray = StackImages(**self.imParameters)
				self.imArray_collector[Bx] = imArray
			
			# Plot the subplots
			if n==0:
				nImages = ((self.imParameters['lastIm'] - self.imParameters['firstIm'])*2)
				frame_colors = get_colors(nImages,'magma',norm=True,visualization_library=visualization_library)
				bx_colors = self.get_color_sequence(colormap, visualization_library)
				self.frame_colors = frame_colors

			i, j = np.int(np.floor(n/cols)), n%cols
			if visualization_library == 'mpl':
				_fig1, _fig2, _fig3 = self.fig1, self.fig2, self.fig3
				_fig4, _fig5, _fig6 = self.fig4, self.fig5, self.fig6
				_fig7, = self.fig7
				_ax1, _ax2, _ax3 = self.axs1[i,j], self.axs2[i,j], self.axs3[i,j]
				_ax4, _ax5, _ax6 = self.axs4[i,j], self.axs5[i,j], self.axs6[i,j]
				_ax7 = self.axs7

			elif visualization_library == 'bokeh':
				_fig1, _fig2, _fig3, _fig4, _fig5, _fig6, _fig7 = n_bokeh_plots * [None]
				_ax1, _ax2, _ax3, _ax4, _ax5, _ax6, _ax7 = n_bokeh_plots * [None]
				_ax7 = [_ax7, _ax7]
				
			# Figure 1 : color Image of DW motion
			figs[1] = imArray.showColorImage(self.gray_threshold, palette=colormap, plotHist=None, 
											plot_contours=False, fig=_fig1, ax=_ax1, title=title, noSwitchColor='black')
			# Figure 2 : plot the histogram
			figs[2] = imArray.plotHistogram(imArray._switchTimesOverThreshold,fill_color=frame_colors,fig=_fig2, ax=_ax2,title=title,ylabel=None)
			# Calculate the contours
			figs[3] = imArray.plotContours(lines_color='black',remove_bordering=True,
											plot_centers_of_mass=True, color_center_of_mass=frame_colors[n], fig=_fig3, ax=_ax3, title=title)
			figs[4] = imArray.plotContours(lines_color='black', remove_bordering=True,reference='center_of_mass',rescale_area=True,
											fig=_fig4, ax=_ax4, title=title)

			# ################################################ 
			self.all_contours[Bx] = imArray.contours
			self.all_centers[Bx] = imArray.centers_of_mass
			# Center of mass of the initial domain 
			center = imArray.centers_of_mass[0] 
			#Plot the central domain
			cnts0 = imArray.contours[0]

			#Plot the external contour
			lastKey = sorted(imArray.contours.keys())[-1]
			cnts = imArray.contours[lastKey]
			xcnts, ycnts = cnts[:,1], cnts[:,0]
			self.xcnts, self.ycnts = cnts[:,1], cnts[:,0]
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

			limits = np.floor(xmin/100.)*100, np.ceil(xmax/100.)*100, np.ceil(ymax/100.)*100, np.floor(ymin/100.)*100
			step_angle = np.pi/10
			title_Bx = r"$B_x (%s)$" % (self.Bx_unit)
			if visualization_library == 'mpl':
				for axs in [self.axs1,self.axs3]:
					axs[i,j].plot(xcnts,ycnts,c=bx_colors[n], lw=1, label=label)
			elif visualization_library == 'bokeh':
				for axs in [figs[1], figs[3]]:
					(x0, y0), (x1, y1) = self.imParameters['imCrop']
					H = y1 - y0
					axs.line(xcnts, H-ycnts, color=bx_colors[n],line_width=2,legend_label=str(label))


			# Plot dispacements in polar coordinates from the center
			fig5, theta, r, frames = polar.plot_displacement(imArray.contours,origin=center,reference='center',
									swope_xy=True,fig=_fig5,ax=_ax5,title=title,step_in_frames=self.step_in_frames, 
									visualization_library=visualization_library)
			# Plot dispacements in polar coordinates from the nucleated domain
			fig6, last_theta, last_r, frames = polar.plot_displacement(imArray.contours,origin=center,reference='nucleated_domain',
									swope_xy=True,fig=_fig6,ax=_ax6,title=title,step_in_frames=self.step_in_frames,
									visualization_library=visualization_library)
					
			if visualization_library == 'bokeh':
				figs[5], figs[6] = fig5, fig6
				figs[5].line(theta/np.pi*180, r, color=bx_colors[n], line_width=2)
				figs[6].line(last_theta/np.pi*180, last_r, color=bx_colors[n], line_width=2)
			elif visualization_library == 'mpl':
				self.axs5[i,j].plot(theta/np.pi*180,r,c=bx_colors[n],lw=2)
				self.axs6[i,j].plot(last_theta/np.pi*180, last_r, c=bx_colors[n], lw=2)


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
			fig7a = polar.plot_mean_velocity(thetas,v_mean,v_error,fig=_fig7,ax=_ax7[0],
									title=title,color=bx_colors[n], visualization_library=visualization_library)
			fig7b = polar.plot_mean_velocity(thetas,v_mean,v_error,fig=_fig7,ax=_ax7[1],
										title="",label=label,color=bx_colors[n], visualization_library=visualization_library)
			
			if visualization_library == 'bokeh':
				for k in range(1, n_bokeh_plots):
					self.plots[k].append(figs[k])


		# if j != cols-1 and visualization_library == 'bokeh':
		# 	for n in range(1,n_bokeh_plots+1):
		# 		for k in range(j+1, cols):
		# 			row[n].append(None)
		# 		plots[n].append(row[n])
		# Plot velocities at different angles
		v = self.velocities
		v_err = self.velocities_error
		v.index = v.index * 180 / np.pi
		v_err.index = v_err.index * 180 / np.pi
		i_shift = int((n_alphas - 1) / 2)
		cl = get_colors(i_shift+1,'magma',True,visualization_library=visualization_library)[1:]
		xlabel = r"$B_x$ ({})".format(self.Bx_unit)
		ylabel = "microns/s"

		for i in range(i_shift):
			col = 0*(i!=0) + 1*(i==0)
			label = "{} deg".format(v.index[i])
			if visualization_library == 'mpl':
				self.axs7[col].errorbar(v.columns,v.iloc[i],v_err.iloc[i],fmt='--o',c=cl[i],label=label)
				self.axs7[col].errorbar(v.columns,v.iloc[i+i_shift],v_err.iloc[i+i_shift],fmt='-o',c=cl[i],label=label)
			elif visualization_library == 'bokeh':
				labels = xlabel, ylabel, label
				fig7a = mkb.plot_errorbar(v.columns, v.iloc[i], v_err.iloc[i], labels, color=cl[i], size=5, fig=fig7a) 
				label = "{} deg".format(v.index[i+i_shift])
				figt7b = mkb.plot_errorbar(v.columns, v.iloc[i], v_err.iloc[i], labels, color=cl[i], size=5, fig=fig7b) 
		
		if visualization_library == 'mpl':
			for i in range(2):
				self.axs7[i].set_ylabel(ylabel)
				self.axs7[i].set_xlabel(xlabel)
				l1 = self.axs7[i].legend(fontsize=12,title="Angle (deg)",
					bbox_to_anchor=(-.05, 1), loc=1, borderaxespad=0.)
				l1.set_draggable(True)
		elif visualization_library == 'bokeh':
			# Add the legend
			self.plots[7] = [fig7a, fig7b]

		# # Last plot
		# ax = axs[rows-1,cols-1]
		# ax.plot(center[1],center[0],'ko')
		# polar.plot_rays((center[1],center[0]),step_angle,ax,limits)
		# ax.axis(limits)
		# #ax.set_aspect('equal')
		# l1 = ax.legend(fontsize=12,title=title_Bx,
		# 				bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.)
		# l1.set_draggable(True)

		# ax = self.axs7[rows-1,cols-1]
		# l1 = ax.legend(fontsize=12,title=title_Bx,numpoints=1,
		#  	bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.)
		# l1.set_draggable(True)

		# for axs in [self.axs4,self.axs5]:
		# 	ax = axs[rows-1,cols-1]
		# 	ax.grid(True)
		# 	ax.set_xlabel("angle (deg)")
		# 	l1 = ax.legend(fontsize=12,title=title_Bx,ncol=1,
		# 		bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.)
		# 	l1.set_draggable(True)
		# self.axs4[rows-1,cols-1].set_title("Last contour")
		# self.axs4[rows-1,cols-1].set_title("Velocity")

		# #self.axs4[rows-1,cols-1].plot(theta/np.pi*180,r,c=colors[n],lw=2,label=label)
		# self.axs5[rows-1,cols-1].plot(theta/np.pi*180,last_r,c=colors[n],lw=2,label=label)
			
		# for fig in self.figs:
		# 	fig.suptitle(self.full_title,fontsize='xx-large')



		if isPlot:
			if visualization_library == 'mpl':
				plt.show()
			elif visualization_library == 'bokeh':
				
				gridplots = []
				for i in range(1,n_bokeh_plots+1):
					_gr = gridplot(self.plots[i], ncols=cols, sizing_mode='scale_both',merge_tools=True)
					gridplots.append(_gr)
				self.fig1, self.fig2, self.fig3, self.fig4, self.fig5, self.fig6, self.fig7 = gridplots
		self.is_plot_all_figures_done = True
				

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
		l1.set_draggable(True)


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

	creep_data = Creep(Bz, iniFilepath)
	creep_data.plot_results()