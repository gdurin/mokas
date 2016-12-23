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
import skimage
import skimage.io as im_io
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist
import bilateralFilter2 as blf
import gaussianFilter as gf
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

    resolution: [8,16] bits
        Image resolution (8 bits default)
    """
    def __init__(self, root_dir, pattern, firstIm=0, lastIm=-1, resolution=16,
                is_hist_equalization=False, resize_factor=None, crop=None, 
                filtering=None, sigma=None): 
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
        #print(resolution)
        if resolution == 8:
            self.resolution = np.int8
        elif resolution == 16:
            self.resolution = np.int16
        elif resolution == 32:
            self.resolution = np.int32
        #print(self.resolution)
        self.is_hist_equalization = is_hist_equalization
        #print(self.mode)
        if self.mode == 'pattern':
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
            return 'pattern'
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
        image = im_io.imread(f).astype(self.resolution)
        if self.filtering:
            return self._image_filter(image)
        else:
            return image

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
            max_gray_level = tif.micromanager_metadata['display_settings'][0]['Max']
            bit_depth = tif.micromanager_metadata['summary']['BitDepth']
        self.images = self.images.astype(self.resolution)
        try:
            assert self.images.shape == (frames, height, width)
            print("TIFF Images loaded...")
        except AssertionError:
            n, xdim, ydim = self.images.shape
            print("Warning: n. of loaded frames is less than expeced %i/%i" % (n, frames))
            print("Original file: (", frames, height, width, ")")
            print("Loaded file", self.images.shape)
            frames = n

            #sys.exit()
        # Check if the gray range is 2**BitDepth, otherwise do histogram equalization
        self.is_hist_equalization = True
        if max_gray_level != 2**bit_depth - 1 and self.is_hist_equalization==False:
            print("The gray level range %i is smaller than the expected %i") % (max_gray_level, 2**bit_depth)
            print("You could perform an histogram equalization")
        if max_gray_level <= .8*2**bit_depth:
            self.is_hist_equalization = True
            print("The gray level range %i is way smaller than the expected %i!") % (max_gray_level, 2**bit_depth)
            print("I am performing an histogram equalization")
        if self.is_hist_equalization:
            print("Equalizing...")
            self.images = self._image_equalize_hist(self.images, full_sequence=True, bit_depth=bit_depth)
            factor = 2**bit_depth/float(max_gray_level)
            print("Done")
        self.images, self.imageNumbers = self._set_limits(self.images, frames)
        try:
            assert len(self.images) == len(self.imageNumbers)
            print("Checking length... OK; there are {} images".format(len(self.imageNumbers)))
        except AssertionError as e:
            print(e)
            print("Checking lenght... Failed")
            print("n. of images: %i") % len(self.images)
            print("Len of imageNumbers: %i") % len(self.imageNumbers)
        # Filtering
        if self.filtering:
            if self.filtering == 'bilateral':
                self.images_raw = np.copy(self.images)
                self.images = self._image_filter(self.images_raw)
            elif self.filtering == 'gauss_parallel':
                #self.images_raw = np.copy(self.images)
                self.images = self._image_filter(self.images)#_raw)
            else:
                for n, image in enumerate(self.images):
                    self.images[n] = self._image_filter(image)
       
    def _image_crop(self, crop_limits):
        """
        crop limits are in the image reference frame (not of array)
        has to be a list of two pixels,
        i.e. [crop_upper_left_pixel,crop_lower_right_pixel]
        """
        n, rows, cols = self.images.shape
        [(col_min,row_min),(col_max,row_max)] = crop_limits
        #xmin, xmax, ymin, ymax = crop_limits
        self.images = self.images[:, row_min : row_max, col_min : col_max]

    def _image_resize(self, resize_factor):
        print("Resize is not available in tiff images, sorry")
        print("Do you really need it?")
        return None

    def _image_filter(self, image):
        if self.filtering == 'bilateral':
            delta = int(np.std(self.images.flatten())*0.1)
            radius, repetitions = 5, 3 #small radius, delta half of jump (for big delta, far colors are closer and get mixed, repetitions can be many if radius is small)
            out = blf.bilateralFilter(image, radius, self.sigma, delta, repetitions, device=0)
            return out
        elif self.filtering == 'gauss_parallel':
            radius = 5
            out = gf.gaussianFilter(image, radius, self.sigma, device=0)
            return out

        else:
            return filters[self.filtering](image, self.sigma)

    def _image_equalize_hist(self, images, full_sequence=True, bit_depth=12):
        # Do histogram equalization (experimental)
        if full_sequence:
            images = equalize_hist(images, nbins=(2**bit_depth))*(2**bit_depth)
        else:
            print("Do histogram equalization")
            for i, im in enumerate(images):
                eqh = equalize_hist(im, nbins=2**bit_depth)*2**bit_depth
                images[i] = eqh.astype(self.resolution)
        return images


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
        if self.mode != 'pattern':          
            if self.firstIm != 0 or self.lastIm != -1:
                self.images = self.images[self.firstIm : self.lastIm + 1]
                self.imageNumbers = self.imageNumbers[self.firstIm : self.lastIm + 1]
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
    filtering=None, sigma=None, subtract=None):
    """
    subtract: int or None
        Subtract image # as background
    """
    im = Images(root_dir=root_dir, pattern=pattern, firstIm=firstIm, lastIm=lastIm, 
        resize_factor=resize_factor, crop=crop, filtering=filtering, sigma=sigma)
    images, imageNumbers = im.collector()
    if subtract is not None:
        # TODO: fix the way the gray level is renormalized
        # This is too rude!
        images = images[subtract+1:] - images[subtract] + np.int(np.mean(images[subtract]))
        imageNumbers = imageNumbers[subtract+1:]
    assert len(images) == len(imageNumbers)
    return images, imageNumbers


if __name__ == "__main__":
    filename = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/Irr_800He/Irr_400uC_8e8He+/05_Irr_8e8He+_0.1A_2fps/05_Irr_8e8He+_0.1A_2fps_MMStack_Pos0.ome.tif"
    root_dir, pattern = os.path.split(filename)
    #root_dir = "/home/gf/Meas/Creep/Alex/PtCoPt_simm/run6/imgs"
    #pattern = "img*.tif"
    im_crop = None  
    #im_crop = (876,1117,0,1040)
    #filtering = 'gauss_parallel'
    filtering = None
    sigma = 2
    out, n = images2array(root_dir, pattern, filtering=filtering, sigma=sigma, crop=im_crop, subtract=None)
    print(out.shape)
    #fout = "exp_40mV_20s_21.pkl"
    #pickle.dump(out, fout)
