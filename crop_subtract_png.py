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


def save_img(array, filename, saveDir):
	img = Image.fromarray(array.astype(np.uint8))
	base, ext = os.path.splitext(os.path.basename(file))
	savePath = os.path.join(saveDir, base + ".png")
	img.save(savePath)


if __name__ == "__main__":

	rootDir = "/home/mokas/Meas/Creep/MeinzMAS185/run2/"
	imagesDir = os.path.join(rootDir, "MeinzMAS185-000nOe-5.000V-796.0s_1")
	saveDir = os.path.join(rootDir, "cropSubtractPng (tests)")
	pattern = os.path.join(imagesDir, "seq1_*.tif")
	backgroundFile = os.path.join(imagesDir, "seq1_00001.tif")

		  # x1   y1   x2   y2
	crop = (294, 321, 774, 760)

	n_average = 10 #Number of images to take for the moving average. Put 1 for no averaging
	# TODO : True doesn't work (Why ?)
	keep_all_images = False #If False, for each n_average image we save only one image : the average of the last n_average images

	inverted = True

	#None for no filtering
	gauss_radius = 1.5

	#(0, 255) for nothing
	gray_level = (110, 145)




	if not os.path.exists(saveDir):
		os.makedirs(saveDir)

	filenames = glob.glob(os.path.join(imagesDir, pattern))

	if(backgroundFile is not None):
		backgroundArray = io.imread(backgroundFile)
		filenames.remove(backgroundFile)

		if(crop is not None):
			ymin, xmin, ymax, xmax = crop
			backgroundArray = backgroundArray[xmin:xmax, ymin:ymax]
	else:
		backgroundArray = None

	lastArrays = {}

	for i, file in enumerate(sorted(filenames)):

		array = io.imread(file).astype(np.intc)
		array = process_image(array, crop=crop, backgroundArray=backgroundArray,
							  inverted=inverted, gauss_radius=gauss_radius, gray_level=gray_level)

		lastArrays[i%n_average] = array

		if(keep_all_images or i%n_average == n_average-1):
			array = moving_average(lastArrays)
			save_img(array, file, saveDir)