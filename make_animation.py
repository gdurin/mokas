from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tifffile
import skimage.exposure
import scipy.ndimage.filters as filters

def ani_frame(images,firstIm,lastImage,subtract_first=True,filtering=None, 
        sigma=1.,out_name='demo.mp4',fps=10,dpi=100):
    """
    images must be a numpy array
    """
    n_images, dimX, dimY = images.shape
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if filtering:
        print("Filtering...")
        #image = skimage.exposure.equalize_hist(image)
        images = filters.gaussian_filter(images, sigma)
        print("Done.")
    if subtract_first:
        image = images[firstIm+1] - images[firstIm]
        average = np.int(np.mean(image))
        firstIm += 1 
    else:
        image = images[firstIm]
    im = ax.imshow(image,cmap='gray',interpolation='nearest')
    #im.set_clim([0,1])
    #fig.set_size_inches([5,5])

    #plt.tight_layout()

    def update_img(n):
        try:
            if subtract_first:
                image = images[n] - images[firstIm] + average
            else:
                image = images[n]
            #print(".",end="")
            im.set_data(image)
            #im.set_array(image)
        except:
            print(n)
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,range(firstIm,lastImage),
        interval=10, blit=False)
    writer = animation.writers['ffmpeg'](fps=fps)
    ani.save(out_name,writer=writer,dpi=dpi)
    plt.close(fig.number)
    return ani

if __name__ == "__main__":
    
    if True:
        #mainDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated"
        #mainDir = "/home/gf/Meas/Creep/CoFeB/Film/Non-irradiated/Moon/run6"
        mainDir = "/home/gf/Meas/Creep/CoFeB/Wires/Irradiated/run1_2"
        for subDir in sorted(os.listdir(mainDir)):
            s = os.path.join(mainDir, subDir)
            for filename in os.listdir(s):
                basename, extension = os.path.splitext(filename)
                if extension == '.mp4':
                    break
                if extension == '.tif':
                    print(filename)
                    f = os.path.join(s, filename)
                    try:
                        with tifffile.TiffFile(f) as tif:
                            im = tif.asarray()
                        # Need to transform into a int16
                    except:
                        print("There is a problem with the tif file, skipping")
                        break
                    im = np.asarray(im, dtype=np.int16)
                    fout = os.path.join(s, basename+".mp4")
                    lastImage, dimX, dimY = im.shape
                    ani_frame(im,1,lastImage,subtract_first=True,filtering=True,
                        sigma=1.,dpi=600,out_name=fout)
    else:
        mainDir = "/home/gf/Meas/Creep/CoFeB/Film/Irradiated/07_irradiatedFilm_0.20A_10fps"
        f = "07_irradiatedFilm_0.20A_10fps_MMStack_Pos0.ome.tif"
        fout = "07_irradiatedFilm_0.20A_10fps_MMStack_Pos0.ome.mp4"
        filename = os.path.join(mainDir, f)
        print(filename)
        with tifffile.TiffFile(filename) as tif:
            im = tif.asarray()
        im = np.asarray(im, dtype=np.int16)
        fout = os.path.join(mainDir, fout)
        lastImage, dimX, dimY = im.shape
        ani_frame(im,1,lastImage,subtract_first=True,dpi=600, filtering=True,
                        out_name=fout)
