from __future__ import print_function
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from visualBarkh import StackImages
from run_creep import get_colors


p2p = 3 # Pixel to pixel (linear) distance for cluster detection
NN = 2*p2p + 1
structure = np.ones((NN,NN))

class Wires:
    def __init__(self, rootDir, subdir_pattern, filename_suffix, imParameters,
                threshold, experiments=None):
        self.rootDir = rootDir
        self.sub_dirs = sorted(glob.glob1(rootDir, subdir_pattern))
        self.filenames = [d+filename_suffix for d in self.sub_dirs]
        if experiments is None:
            self.n_experiments = len(self.sub_dirs)
            self.experiments = range(self.sub_dirs)
        else:
            self.n_experiments = len(experiments)
            self.experiments = experiments
        self.imParameters = imParameters
        self.threshold = threshold
        print(self.filenames[0])
        self.full_title = ", ".join(self.filenames[0].split("_")[1:4])
       

    def plot_results(self):
        """
        Plot the different images for creep in wires
        """
        # Prepare to plot
        plt.close("all")
        self.figs = []
        self.imArray_collector = {}
        print("Preparing plots",end="")
        self.fig1, self.axs1 = plt.subplots(1,self.n_experiments,sharex=True, sharey=True) # ColorImages of the wires
        self.figs.append(self.fig1)
        print(".",end="")
        self.fig2, self.axs2 = plt.subplots(self.n_experiments,1) # Histograms
        self.figs.append(self.fig2)
        print(".",end="")
        self.fig3, self.axs3 = plt.subplots(1,self.n_experiments,sharex=True, sharey=True) # Contours
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
            imArray = StackImages(**self.imParameters)
            self.imArray_collector[trial] = imArray
            imArray.useKernel = 'step'
            imArray.kernelSign = -1
            imArray.boundary = None
            imArray.structure = structure
            imArray.showColorImage(self.threshold,palette=pColor,plot_contours=False,plotHist=None,
                                   fig=self.fig1,ax=self.axs1[n],title=title,noSwitchColor='black')
            imArray.plotHistogram(imArray._switchTimesOverThreshold,
                                    fig=self.fig2,ax=self.axs2[n],title=title,ylabel=None)
            imArray.find_contours(lines_color='k',remove_bordering=True,plot_centers_of_mass=False,
                                     invert_y_axis=True, plot_rays=False,
                                     fig=self.fig3,ax=self.axs3[n],title=title)
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
        imParameters['imCrop'] = (0,1392,0,1040)
        imParameters['pattern'] = "02_nonirradiatedFilm_0.14A_MMStack_Pos0.ome.tif"
        imParameters['firstIm'] = 1
        imParameters['lastIm'] = 250
        imParameters['filtering'] = 'gauss'
        #imParameters['filtering'] = None
        imParameters['sigma'] = 1.
        imParameters['resize_factor'] = None
        threshold = 30
    elif choice == 'non_irr':
        set_current = ["0.20","0.22","0.24"][0]
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/nonirrad wire/"
        if not os.path.isdir(rootDir):
            print("Chech the path")
            sys.exit()
        subdir_pattern = "*_nonirradiatedwires_%sA_10fps"  % set_current

        #imParameters['imCrop'] = (0,1392,0,1040)
        imParameters['imCrop'] = (1000,1392,100,1040)
        #imParameters['imCrop'] = (876,1117,0,1040)
        filename_suffix = "_MMStack_Pos0.ome.tif"
        #imParameters['pattern'] = "01_irradiatedwires_%sA_10fps_MMStack_Pos0.ome.tif" % set_current
        imParameters['firstIm'] = 1
        imParameters['lastIm'] = 300
        imParameters['filtering'] = 'gauss'
        #imParameters['filtering'] = None
        imParameters['sigma'] = 1.
        imParameters['resize_factor'] = None
        threshold = 20
        wires = Wires(rootDir, subdir_pattern, filename_suffix, imParameters,threshold, experiments=range(6))
        wires.plot_results()
    
    elif choice == 'irr':
        set_current = ["0.19","0.21","0.23"][0]
        rootDir = "/home/gf/Meas/Creep/CoFeB/Wires/Irradiated/run1_2/"
        subdir_pattern = "*_irradiatedwires_%sA_10fps"  % set_current

        #imParameters['imCrop'] = (0,1392,0,1040)
        imParameters['imCrop'] = (876,1117,0,1040)
        filename_suffix = "_MMStack_Pos0.ome.tif"
        #imParameters['pattern'] = "01_irradiatedwires_%sA_10fps_MMStack_Pos0.ome.tif" % set_current
        imParameters['firstIm'] = 30
        imParameters['lastIm'] = 300
        imParameters['filtering'] = 'gauss'
        #imParameters['filtering'] = None
        imParameters['sigma'] = 1.
        imParameters['resize_factor'] = None
        threshold = 20
        wires = Wires(rootDir, subdir_pattern, filename_suffix, imParameters,threshold, experiments=range(6))
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

