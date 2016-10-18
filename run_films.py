import sys, glob
import numpy as np
import matplotlib.pyplot as plt
import visualBarkh as bk

p2p = 3 # Pixel to pixel (linear) distance for cluster detection
NN = 2*p2p + 1

if __name__ == "__main__":
    imParameters = {}
    n = "02"
    current = "0.16"
    rootDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/" 
    patternDir = "*_irradiatedFilm_%sA_10fps" % (current)
    listofDirectories = sorted(glob.glob1(rootDir, patternDir))
    #imParameters['imCrop'] = (200,1040,500,1390)
    #imParameters['imCrop'] = (270,970,200,950) # good for 01 0.16A
    imParameters['imCrop'] = None
    imParameters['firstIm'] = 0 # Use python convention: start from zero!
    imParameters['lastIm'] = 10
    imParameters['filtering'] = 'gauss'
    #imParameters['filtering'] = None
    imParameters['sigma'] = 2.
    imParameters['subtract'] = 0 # Subtract the first image
    threshold = 8

    imArrays = []
    for directory in listofDirectories[1:]:
        print(directory)
        imParameters['subDirs'] = [rootDir, directory, "", "", ""]
        imParameters['pattern'] = "%s_irradiatedFilm_%sA_10fps_MMStack_Pos0.ome.tif" % (directory[:2], current)

    
        imArray = bk.StackImages(**imParameters)
        imArray.width='all'
        imArray.useKernel = 'both'
        imArray.kernelSign = -1
        #imArray.useKernel = 'zero'
        imArray.imageDir = "Left_to_right"
        imArray.boundary = None
        imArray.structure = np.ones((NN,NN))
        print(imArray)
        imArrays.append(imArray)
        #imArray.showColorImage(threshold,palette=palette,plot_contours=False)
        #imArray.find_contours(lines_color=None, remove_bordering=True, plot_centers_of_mass=False,
        #    plot_rays=False, reference=None,invert_y_axis=True)
            #'center_of_mass')