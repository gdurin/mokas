import sys, os
import numpy as np
import matplotlib.pyplot as plt
import visualBarkh as bk
import cPickle as pickle
import mokas_bubbles as mkb

if __name__ == "__main__":
    plt.close("all")
    imParameters = {}
    print(sys.argv)
    try:
        choice = sys.argv[1]
    except:
	choice = 'bubble'
    if choice == 'bubble':
        k = sys.argv[2]
        print k
        k = str(k).rjust(2,"0")
        rootDir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/%s_Irr_800uC_0.116A" % k
        
        imParameters['pattern'] = "%s_Irr_800uC_0.116A_MMStack_Pos0.ome.tif" % k
        crop_upper_left_pixel, crop_lower_right_pixel = (450,330), (860,750)
        print(imParameters['pattern'])
        imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        #imParameters['imCrop'] = None
        imParameters['firstIm'] = 0 # Use python convention: start from zer$
        imParameters['lastIm'] = -1
        imParameters['filtering'] = 'gauss'
        imParameters['sigma'] = 2
        imParameters['subtract'] = None # Subtract a reference image
        threshold = None
        palette = 'coolwarm'
        erase_small_events_percent = None
        imParameters['resize_factor'] = None
        # Kernel setups: do not touch
        imParameters['kernel_half_width_of_ones'] = 10
        #imParameters['kernel_internal_points'] = 0
        #imParameters['kernel_switch_position'] = "end"
        ##############################
        imParameters['subDirs'] = [rootDir, "", "", "", ""]
        #A possible improvement is that stackImages returns also the threshold value extracting the information when uploading the images
        
        imArray = mkb.Bubbles(**imParameters)
        imArray.showColorImage(threshold=threshold, palette=palette, plot_contours=True, 
            erase_small_events_percent=None)
        #imArray.find_contours(lines_color='k', remove_bordering=True, plot_centers_of_mass=False,
        #    plot_rays=False, reference=None,invert_y_axis=True)
        #'center_of_mass')
        #save_data = raw_input("Save the data?")

        #imArray.pickle_switchMap2D(mainDir=rootDir)
      
