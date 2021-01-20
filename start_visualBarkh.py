# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:06:23 2011

@author: gf
"""
import os
import mokas_stackimages as msi

if __name__ == "__main__":
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/50nm/run2/"
    # Select dir for analysis: TO IMPROVE
    #mainDir = "/home/gf/meas/Baxrkh/Films/CoFe/20nm/run3_50x/down/"
    #mainDir = "/home/gf/meas/Barkh/Films/FeBSi/50nm/run1/down"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run1_20x/down"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run9_20x_5ms"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run10_20x_bin1"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run11_20x_bin1_contrast_diff"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run15_20x_save_to_memory/down"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run32"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/good set 2/run5/"
    #mainDir = "/home/gf/meas/MO/py170/20x/set7"
    #mainDir = "/home/gf/Misure/Alex/Zigzag/samespot/run2"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run22_50x_just_rough/down"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run23_50x_rough_long"
    #mainDir = "/home/gf/meas/Simulation"
    #mainDir = "/media/DATA/meas/MO/CoFe 20 nm/10x/good set 2/run7"
    ##mainDir = "/media/DATA/meas/MO/CoFe 20 nm/5x/set1/run1/"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/10x/good set 2/run8/"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/50nm/run2/"
    rootDir = "/media/DATA/meas/MO/CoFe/50nm/"
    rootDir = "/media/DATA/meas/Barkh/Films/CoFe/50nm/"
    magnification = "20x"
    #runNo, firstImage, lastImage = "run2", 384, 1325    
    #runNo, firstImage, lastImage = "run5", 250, 1050
    #runNo, firstImage, lastImage = "run6", 470, 1431
    runNo, firstImage, lastImage = "run7", 120, 1180
    #runNo, firstImage, lastImage = "run9", 1, 1380
    magnification = "10x"
    runNo, firstImage, lastImage = "run2", 1, 867
    #magnification = "5x"
    #runNo, firstImage, lastImage = "run1", 575, 1060
    
    mainDir = os.path.join(rootDir, magnification, runNo)
    mainDir = "/home/gf/Meas/Creep/PtCoPt/M2/PtCoPt_2-2c-0d-100nOe-0.657V-30.0s_18"
    subDirs = [mainDir, "", "", "", ""]
    firstImage, lastImage = 1, 89
    filtering = 'gauss'
    sigma = 1.5
    pattern = "Data1-*.tif"
    pattern = "filename*.png"
    imCrop = [(160,80),(520,400)]
    gray_threshold = 25
    imArray = msi.StackImages(subDirs,pattern, filtering=filtering, sigma=sigma,\
                             firstIm=firstImage, lastIm=lastImage, imCrop=imCrop)

    imArray.width='small'
    imArray.useKernel = 'step'
