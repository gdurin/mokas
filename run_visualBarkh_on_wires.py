from __future__ import print_function
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import mokas_wires as mkwires
from mokas_colors import get_colors


p2p = 3 # Pixel to pixel (linear) distance for cluster detection
NN = 2*p2p + 1
structure = np.ones((NN,NN))


class RunWires:
    def __init__(self, rootDir, subdir_pattern, filename_suffix):
        self.rootDir = rootDir
        self.sub_dirs = sorted(glob.glob1(rootDir, subdir_pattern))
        self.filenames = [d+filename_suffix for d in self.sub_dirs]
        wire_ini = mkwires.Wires_ini(rootDir, 2)
        self.imParameters = wire_ini.imParameters
        self.experiments = wire_ini.experiments
        self.threshold = wire_ini.threshold
        self.motion = wire_ini.motion
        if self.experiments is None:
            self.n_experiments = len(self.sub_dirs)
            self.experiments = range(self.sub_dirs)
        else:
            self.n_experiments = len(self.experiments)
        print(self.filenames[0])
        self.full_title = ", ".join(self.filenames[0].split("_")[1:4])
       

    def plot_results(self, plot_contours=True):
        """
        Plot the different images for creep in wires
        """
        # Prepare to plot
        plt.close("all")
        self.figs = []
        self.imArray_collector = {}
        self.fig1, self.axs1 = plt.subplots(1, self.n_experiments, sharex=True, sharey=True) # ColorImages of the wires
        self.figs.append(self.fig1)
        self.fig2, self.axs2 = plt.subplots(self.n_experiments, 1, sharey=True) # Histograms
        self.figs.append(self.fig2)
        self.fig3, self.axs3 = plt.subplots(1, self.n_experiments, sharex=True, sharey=True) # Pinning Centers
        self.figs.append(self.fig3)
        

        #for n in range(self.n_experiments):
        for n,experiment in enumerate(self.experiments):
            if n == 0:
                nImages = ((self.imParameters['lastIm'] - self.imParameters['firstIm'])*2)
                pColor = get_colors(nImages,'pastel',norm=True)
            sub_dir = self.sub_dirs[experiment]
            trial = sub_dir[:2]
            title = trial
            self.imParameters['subDirs'] = [self.rootDir, sub_dir, "", "", ""]
            filename = self.filenames[experiment]
            self.imParameters['pattern'] = filename
            print(experiment, filename)
            #imArray = StackImages(**self.imParameters)
            imArray = mkwires.Wires(self.motion, **self.imParameters)
            self.imArray_collector[trial] = imArray
            #imArray.useKernel = 'step'
            #imArray.kernelSign = -1
            #imArray.boundary = None
            #imArray.structure = structure
            imArray.showColorImage(self.threshold,palette=pColor,plot_contours=True,plotHist=None,
                                   fig=self.fig1,ax=self.axs1[n],title=title,noSwitchColor='black')
            imArray.plotHistogram(imArray._switchTimesOverThreshold,
                                    fig=self.fig2,ax=self.axs2[n],title=title,ylabel=None)
            # imArray.find_contours(lines_color='k', remove_bordering=True, plot_centers_of_mass=False,
            #                          invert_y_axis=False, plot_rays=False,
            #                          fig=self.fig3, ax=self.axs3[n], title=title)
            imArray.get_stats_prop()
            self.axs3[n].imshow(imArray.stats_prop['image_corners'])
            if plot_contours:
                for switch in imArray.contours:
                    cnts = imArray.contours[switch]
                    X,Y = cnts[:,1], cnts[:,0]
                    self.axs3[n].plot(X,Y,c='w',antialiased=True,lw=1)

        # Out of the loop
        for fig in self.figs:
            fig.suptitle(self.rootDir,fontsize=30)
        plt.show()

if __name__ == "__main__":
    plt.close("all")
    imParameters = {}
    try:
        choice = sys.argv[1]
    except:
        choice = 'irr'
    if choice == 'half_moon':
        rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Non-irradiated/Half-moon/run3/02_nonirradiatedFilm_0.14A"
        #imParameters['imCrop'] = (200,1040,500,1390)
        #imParameters['imCrop'] = (0,1392,0,1040)
        crop_upper_left_pixel = (0,0)
        crop_lower_right_pixel = (1392,1040)
        imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        #imParameters['imCrop'] = None
        imParameters['pattern'] = "02_nonirradiatedFilm_0.14A_MMStack_Pos0.ome.tif"
        imParameters['firstIm'] = 1
        imParameters['lastIm'] = 250
        imParameters['filtering'] = 'gauss'
        #imParameters['filtering'] = None
        imParameters['sigma'] = 0.5
        imParameters['resize_factor'] = None
        threshold = 30

    elif choice == 'non_irr':
        set_current = ["0.20","0.22","0.24"][0]
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/nonirrad wire/"
        if not os.path.isdir(rootDir):
            print("Chech the path")
            sys.exit()
        subdir_pattern = "*_nonirradiatedwires_%sA_10fps"  % set_current
        filename_suffix = "_MMStack_Pos0.ome.tif"

        # crop_upper_left_pixel = (158,91)
        # crop_lower_right_pixel = (360,1040)     
        # imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        
        # imParameters['firstIm'] = 1
        # imParameters['lastIm'] = 300
        # imParameters['filtering'] = 'gauss'
        # #imParameters['filtering'] = None
        # imParameters['sigma'] = 1.
        # imParameters['resize_factor'] = None
        #threshold = 10
        #experiments = [0,1,2,3,5,6,7,8]
        #experiments = [0,1,2]
        wires = RunWires(rootDir, subdir_pattern, filename_suffix)
        wires.plot_results()
    
    elif choice == 'irr':
        set_current = ["0.19","0.21","0.23"][0]
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Irradiated/run1_2/"
        subdir_pattern = "*_irradiatedwires_%sA_10fps"  % set_current

        #imParameters['imCrop'] = (0,1392,0,1040)
        #imParameters['imCrop'] = (876,1117,0,1040)
        crop_upper_left_pixel = (880,0)
        crop_lower_right_pixel = (1115,1040)
        imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        #imParameters['imCrop'] = None
        filename_suffix = "_MMStack_Pos0.ome.tif"
        #imParameters['pattern'] = "01_irradiatedwires_%sA_10fps_MMStack_Pos0.ome.tif" % set_current
        imParameters['firstIm'] = 1
        imParameters['lastIm'] = 240
        imParameters['filtering'] = 'gauss'
        #imParameters['filtering'] = None
        imParameters['sigma'] = 1.5
        threshold = 10
        wires = RunWires(rootDir, subdir_pattern, filename_suffix, imParameters, threshold, experiments=range(2))
        wires.plot_results()
    else:
        print("Check the path!")
        sys.exit()

        
        

        # imParameters['subDirs'] = [rootDir, "", "", "", ""]
        # imArray = bk.StackImages(**imParameters)
        # imArray.width='all'
        # imArray.useKernel = 'both'
        # imArray.kernelSign = -1
        # #imArray.useKernel = 'zero'
        # imArray.imageDir = "Left_to_right"
        # imArray.boundary = None
        
        # imArray.structure = np.ones((NN,NN))
        # imArray.showColorImage(threshold,palette='pastel',plot_contours=False)
        # #imArray.find_contours(lines_color='k',remove_bordering=True, plot_centers_of_mass=True,reference=None)
        #     #'center_of_mass')

