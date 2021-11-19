import glob
import os
import skimage.io as io
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def process_image(array, crop=None, backgroundArray=None, inverted=False, gauss_radius=None, gray_level=(0, 255)):    
	gray_min, gray_max = gray_level
	
	if(crop is not None):
		ymin, xmin, ymax, xmax = crop
		array = array[xmin:xmax, ymin:ymax]

	if(backgroundArray is not None):
		array = array - backgroundArray
		array = ((array-array.min()) / (array.max()-array.min()))*255
	
	a = 255/(gray_max - gray_min)
	b = -a * gray_min
	array = a * array + b
	
	array[array < 0] = 0
	array[array > 255] = 255
	
		
	if(inverted):
		array = 255 - array
		
	if(gauss_radius is not None):
		array = gaussian_filter(array, gauss_radius)

	return array
	
	
def moving_average(dictArray):
	arrayAverage = sum(dictArray.values())
	return arrayAverage/len(dictArray)


def save_img(array, i, filename, saveDir, renameImages=None):
	img = Image.fromarray(array.astype(np.uint8))
	if renameImages is None:
		base, ext = os.path.splitext(os.path.basename(file))
		savePath = os.path.join(saveDir, base + ".png")
	else:
		base, ext = renameImages
		savePath = os.path.join(saveDir, base + "%05d" % (i, ) + ext)
	img.save(savePath)

def images_processing(imagesDir, pattern, saveDir,
					backgroundFile=None, crop=None, inverted=False,
					gauss_radius=None, gray_level=(0, 255),
					n_average=1, keep_all_images=False, renameImages=None):
	if not os.path.exists(saveDir):
		os.makedirs(saveDir)

	filenames = glob.glob(os.path.join(imagesDir, pattern))

	if(backgroundFile is not None):
		backgroundArray = io.imread(backgroundFile)
		if backgroundFile in filenames:
			filenames.remove(backgroundFile)

		if(crop is not None):
			ymin, xmin, ymax, xmax = crop
			backgroundArray = backgroundArray[xmin:xmax, ymin:ymax]
	else:
		backgroundArray = None

	lastArrays = {}
	count = 0

	for i, file in enumerate(sorted(filenames)):

		array = io.imread(file).astype(np.intc)
		array = process_image(array, crop=crop, backgroundArray=backgroundArray,
							  inverted=inverted, gauss_radius=gauss_radius, gray_level=gray_level)

		lastArrays[i%n_average] = array

		if(keep_all_images or i%n_average == n_average-1):
			array = moving_average(lastArrays)
			save_img(array, count, file, saveDir, renameImages)
			count += 1


if __name__ == "__main__":

	rootDir = "/home/mokas/Meas/Creep/MeinzMAS185/run2/"
	imagesDir = os.path.join(rootDir, "MeinzMAS185-000nOe-5.000V-796.0s_1")
	saveDir = os.path.join(rootDir, "cropSubtractPng (tests)")
	pattern = os.path.join(imagesDir, "seq1_*.tif")


	backgroundFile = os.path.join(imagesDir, "seq1_00001.tif") # None for no backgroubnd subtract
		  # x1   y1   x2   y2
	crop = (294, 321, 774, 760) #None for no crop

	n_average = 10 #Number of images to take for the moving average. Set to 1 for no averaging
	# TODO : True doesn't work (Why ?)
	keep_all_images = False #If False, for each n_average image we save only one image : the average of the last n_average images

	inverted = False

	#None for no filtering
	gauss_radius = 1.5

	#(0, 255) for nothing (8 bits images) (or (0, 65535) for 16 bits images)
	gray_level = (110, 145)

	#None for no renaming
	renameImages = ("seq1_", ".png")

	images_processing(imagesDir, pattern, saveDir,
					backgroundFile=backgroundFile, crop=crop, inverted=inverted,
					gauss_radius=gauss_radius, gray_level=gray_level,
					n_average=n_average, keep_all_images=keep_all_images, renameImages=renameImages)