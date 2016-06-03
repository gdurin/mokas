# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:06:32 2013

@author: gf
"""

import os, re, glob, sys
import rof
import numpy as np
import Image
import scipy.misc as misc

def denoise(im):
    U,T = rof.denoise(im,im)
    return np.asarray(U, dtype='int32')
    

import skimage.io as im_io

class Imread_convert():

    def __call__(self, f):
        im = np.array(Image.open(f), dtype='int32')
        #            imageList = list(im.getdata())
        #            sizeX, sizeY = im.size
        #            return np.asanyarray(imageList).reshape(sizeY, sizeX)
        return im


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    
mainDir, pattern = "/media/DATA/meas/Barkh/Films/NiO-Fe/NiO80/NiO80 run19 10x pinLin 10iter/", "Data1-*.tif"

s = "(%s|%s)" % tuple(pattern.split("*"))
patternCompiled = re.compile(s)
# Load all the image filenames
imageFileNames = glob.glob1(mainDir, pattern)
# Sort it with natural keys
imageFileNames.sort(key=natural_key)

if not len(imageFileNames):
    print "ERROR, no images in %s" % mainDir
    sys.exit()
else:
    print "Found %d images in %s" % (len(imageFileNames), mainDir)
    
# Prepare the rof subdir
rofDir = os.path.join(mainDir, 'rof')
if not os.path.exists(rofDir):
    os.mkdir(rofDir)

    
imread_convert = Imread_convert()
# Load the images
print "Loading images: "
load_pattern = [os.path.join(mainDir,ifn) for ifn in imageFileNames]

imageCollection = im_io.ImageCollection(load_pattern, load_func=imread_convert)

for i, im in enumerate(imageCollection):
    print "Image n. %i" % i
    U = denoise(im)
    fileName = imageFileNames[i]
    fileName = os.path.join(mainDir, 'rof', fileName)
    q = misc.toimage(U)
    q.save(fileName)
    
