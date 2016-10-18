# file video
# to load numpy arrays from a video
import sys
import os
import glob
import re
import numpy as np
import cv2
import pickle
import tifffile
import scipy.ndimage as nd
import scipy.signal as signal
import skimage.io as im_io
from PIL import Image


filters = {'gauss': nd.gaussian_filter, 'fouriergauss': nd.fourier_gaussian,\
           'median': nd.median_filter, 'wiener': signal.wiener, 'rof':None, 'binary': None}


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

class Images:
    """
    load load images using a pattern (i.e. image*.png) or a movie filename 

    Parameters:
    ----------------
    root_ir : string
        Directory of the image file(s)

    pattern : string
        Pattern of the input image files,
        as for instance "Data1-*.tif"
        or filename of a movie (.avi)

    firstImage, lastIm : int, opt
       first and last image (included) to be loaded
       These numbers refer to the numbers of the filenames,
       or to the number of the frames, starting from 0
    """
    def __init__(self, root_dir, pattern, firstIm=0, lastIm=-1,
                resize_factor=None, crop=None, filtering=None, sigma=None): 
        """
        initialization 
        """
        self.root_dir = root_dir
        self.pattern = pattern
        self.filename = os.path.join(root_dir, pattern)
        self.firstIm = firstIm
        self.lastIm = lastIm
        self.resize_factor = resize_factor
        self.crop = crop
        self.filtering = filtering
        self.sigma = sigma
        self.mode = self._set_mode()
        #print(self.mode)
        if self.mode == 'pat':
            self.from_type = self._from_pattern
        elif self.mode == 'avi':
            self.from_type = self._from_avi
        elif self.mode == 'tif':
            self.from_type = self._from_tif
        else:
            print("Mode not available")
            sys.exit()

    def _set_mode(self):
        if "*" in self.pattern:
            return 'pat'
        else:
            basename, extension = os.path.splitext(self.pattern)
            return extension[1:]

    def _set_limits(self, images, n):
        """
        load the images within the firstIm and lastIm
        chosen by the user
        n is the # of images
        """
        if self.firstIm == None:
            self.firstIm = 0
        if self.lastIm == -1 or self.lastIm > n-1:
            self.lastIm = n - 1
        images = images[self.firstIm:self.lastIm+1]
        imageNumbers = range(self.firstIm, self.lastIm+1)
        return images, imageNumbers

    def _imread_convert(self,f):
        """
        function to read and filter images
        """
        if self.filtering:
            image = im_io.imread(f).astype(np.int16)
            return filters[self.filtering](image, self.sigma)
        else:
            return imread(f).astype(np.int16)

    def _from_pattern(self):
        """
        load images from a pattern
        """
        self.imageNumbers, imageFileNames, imageMode = self._image_names()
        #imread_convert = Imread_convert(imageMode)
        # Load the images
        print("Loading images: ")
        load_pattern = [os.path.join(self.root_dir, ifn) for ifn in imageFileNames]
        # Collect the images
        self.imageCollection = im_io.ImageCollection(load_pattern, load_func=self._imread_convert)
        # Concatenate and return
        self.images = im_io.concatenate_images(self.imageCollection)
        print("Done...")
        return

    def _from_avi(self):
        is_initialized = False
        k = 0
        cap = cv2.VideoCapture(self.filename)
        while(cap.isOpened()):
            print(k)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.filtering:
                gray = self._image_filter(gray)
            gray = gray[np.newaxis,...]
            if not is_initialized:
                self.images = gray
                is_initialized = True
            else:   
                self.images = np.vstack((self.images,gray)) 
            k += 1
        self.imageNumbers = range(k)
        cap.release()

    def _from_tif(self):
        with tifffile.TiffFile(self.filename) as tif:
            frames = tif.micromanager_metadata['summary']['Frames']
            height = tif.micromanager_metadata['summary']['Height']
            width = tif.micromanager_metadata['summary']['Width']
            self.images = tif.asarray()
        assert self.images.shape == (frames, height, width)
        self.images = self.images.astype(np.int16)
        self.images, self.imageNumbers = self._set_limits(self.images, frames)
        try:
            assert len(self.images) == len(self.imageNumbers)
        except AssertionError as e:
            print(e)
            print("n. of images: %i") % len(self.images)
            print("Len of imageNumbers: %i") % len(self.imageNumbers)
        # Filtering
        if self.filtering:
            for n, image in enumerate(self.images):
                self.images[n] = self._image_filter(image)

    def _image_crop(self, crop_limits):
        """
        crop limits are in the image reference frame (not of array)
        """
        n, rows, cols = self.images.shape
        xmin, xmax, ymin, ymax = crop_limits
        self.images = self.images[:, rows - ymax : rows - ymin, xmin : xmax]

    def _image_resize(self, resize_factor):
        print("Resize is not available in tiff images, sorry")
        print("Do you really need it?")
        return None

    def _image_filter(self, image):
        return filters[self.filtering](image, self.sigma)

    def _image_names(self):
        """
        get the filenames for a collection of images with a pattern
        """   
        s = "(%s|%s)" % tuple(self.pattern.split("*"))
        patternCompiled = re.compile(s)
        # Load all the image filenames
        imageFileNames = glob.glob1(self.root_dir, self.pattern)
        # Sort it with natural keys
        imageFileNames.sort(key=natural_key)

        if not len(imageFileNames):
            print("ERROR, no images in %s" % self.root_dir)
            sys.exit()
        else:
            print("Found %d images in %s" % (len(imageFileNames), self.root_dir))

        # Search the number of all the images given the pattern above
        if self.pattern[0]!="*":
            image_numbers = [int(patternCompiled.sub("", fn)) for fn in imageFileNames]
        else:
            # To do: solve for a more general case (now works for cecilia files)
            image_numbers = [int(fn[:3]) for fn in imageFileNames]
 
        # Search the indexes of the first and the last images to load
        if self.firstIm is None:
            self.firstIm = image_numbers[0]
        if self.lastIm < 0:
            self.lastIm = len(image_numbers) + self.lastIm + self.firstIm
        try:
            iFirst, iLast = image_numbers.index(self.firstIm), image_numbers.index(self.lastIm)
        except:
            i0, i1 = image_numbers[0], image_numbers[-1]
            out = (i0, i1, self.firstIm, self.lastIm)
            print("Error: range of the images is %s-%s (%s-%s chosen)" % out)
            sys.exit()

        print("First image: %s, Last image: %s" % (imageFileNames[iFirst], imageFileNames[iLast]))
        imageFileNames = imageFileNames[iFirst:iLast + 1]
        
        # Save the list of numbers of the images to be loaded
        imageNumbers = image_numbers[iFirst:iLast + 1]

        # Check the mode of the images
        fname = os.path.join(self.root_dir, imageFileNames[iFirst])
        imageOpen = Image.open(fname)
        imageMode = imageOpen.mode
        
        return imageNumbers, imageFileNames, imageMode

    def collector(self):
        # Upload the images
        self.from_type()
        if self.mode != 'pat':          
            if self.firstIm != 0 or self.lastIm != -1:
                self.images = self.images[self.firstIm : self.lastIm + 1]
        if self.crop is not None:
            print("Original image size: ", self.images.shape)
            self._image_crop(self.crop)
            print("Cropped image size: ", self.images.shape)
        if self.resize_factor:
            self._image_resize(self.resize_factor)
        # if self.filtering:
        #     print("Filtering with %s..." % self.filtering)
        #     self._image_filter(self.filtering, self.sigma)
        try:
            assert len(self.images) == len(self.imageNumbers)
        except AssertionError:
            print("Assertion error")
            print("n. of images: %i") % len(self.images)
            print("Len of imageNumbers: %i") % len(self.imageNumbers)
        return self.images, self.imageNumbers

def images2array(root_dir, pattern, firstIm=0, lastIm=-1, resize_factor=None, crop=None, 
                filtering=None, sigma=None, subtract=None, adjust_gray_level=True):
    """
    subtract: int or None
        Subtract image # as background
    """
    im = Images(root_dir, pattern, firstIm, lastIm, resize_factor, crop, filtering, sigma)
    images, imageNumbers = im.collector()
    if subtract is not None:
        # TODO: fix the way the gray level is renormalized
        # This is too rude!
        images = images[subtract+1:] - images[subtract] + np.mean(images[subtract])
        imageNumbers = imageNumbers[subtract+1:]
    assert len(images) == len(imageNumbers)
    return images, imageNumbers


if __name__ == "__main__":
    #filename = "/home/gf/Meas/Creep/WCoFeB/Const_InPl_Vary_OOP/exp_40mV_20s_21.avi"
    #filename = "/home/gf/Meas/Creep/WCoFeB/Const_InPl_Vary_OOP/exp_50mV_6s_19.avi"
    #filename = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/01_irradiatedFilm_0.16A_10fps/01_irradiatedFilm_0.16A_10fps_MMStack_Pos0.ome.tif"
    filename = "/home/gf/Meas/Creep/CoFeB/Wires/Irradiated/run1_2/01_irradiatedwires_0.19A_10fps/01_irradiatedwires_0.19A_10fps_MMStack_Pos0.ome.tif"
    root_dir, pattern = os.path.split(filename)
    #root_dir = "/home/gf/Meas/Creep/Alex/PtCoPt_simm/run6/imgs"
    #pattern = "img*.tif"
    im_crop = None  
    #im_crop = (876,1117,0,1040)
    filtering = 'gauss'
    filtering = None
    sigma = 1.5
    out, n = images2array(root_dir, pattern, filtering=filtering, sigma=sigma, crop=im_crop, subtract=0)
    print(out.shape)
    #fout = "exp_40mV_20s_21.pkl"
    #pickle.dump(out, fout)
