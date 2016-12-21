import sys
import numpy as np
import scipy.ndimage as nd
import pycuda.tools as tools
from pycuda.gpuarray import to_gpu
from pycuda.compiler import SourceModule
import time
import pycuda.driver as driver
import mokas_gpu as mkGpu
import skimage.filters as filters


def gaussianFilter(stackImages, radius, sigma, device=0):
    startTime = time.time()
    stack32 = np.asarray(stackImages, dtype=np.float32)
    need_mem = 2 * stack32.nbytes +  np.dtype(np.float32).itemsize * (2 * radius + 1)**2
    #free_mem_gpu, total_mem_gpu = driver.mem_get_info()
    current_dev, ctx, (free_mem_gpu, total_mem_gpu) = mkGpu.gpu_init(device)
    print("current device: %s" % current_dev.name())
    print("Total memory to be used: %.2f GB" % (need_mem/1e9))
    print("Total memory of %s: %.2f GB" % (current_dev.name(), total_mem_gpu/1e9))
    free_mem_gpu = 0.9 * total_mem_gpu
    if need_mem < free_mem_gpu:
        #change with parallel filtering
        stack32 = do_gaussianFilter(stack32, radius, sigma)
        stackImages_filtered = stack32
    else:
        nsplit = int(float(need_mem)/free_mem_gpu) + 1
        print("Splitting images in %d parts..." % nsplit)
        stack32s = np.array_split(stack32, nsplit, 0)
        print("Done")
        switchTimes = np.array([])
        switchSteps = np.array([])
        for k, stack32 in enumerate(stack32s):
            print("Calculation split %i" % k)
            #change with parallel filtering
            stack32 = do_gaussianFilter(stack32, radius, sigma)
            if not k:
                stackImages_filtered = stack32
                print(stackImages_filtered.shape)
            else:
                stackImages_filtered = np.vstack((stackImages_filtered, stack32))
                print(stackImages_filtered.shape)
    print('Analysing done in %f seconds' % (time.time()-startTime))
    # Close cuda device
    success = mkGpu.gpu_deinit(current_dev, ctx)
    if not success:
        print("There is a problem with the device %i" % device)
    return stackImages_filtered




def do_gaussianFilter(stackImages, radius, sigma):
    """
    Return a matrix with the positions of a step in a sequence for each pixel

    Parameters:
    ---------------
    stackImages: int32 : 3D Array of images

    useKernel : string
        step = [-1]*5 +[1]*5
        zero = [-1]*5 +[0] + [1]*5
    """
    # Convert to int32
    dim_z, dim_y, dim_x = stackImages.shape
    dim_Z, dim_Y, dim_X = np.int32(stackImages.shape)
    block_X = 256
    block_Y = 1
    grid_X, grid_Y = dim_x*dim_y*dim_z / block_X if dim_x*dim_y*dim_z % block_X==0 else dim_x*dim_y*dim_z / block_X +1 , 1
    print("Print grid dimensions: ", grid_X, grid_Y)
    auxStack = np.zeros((dim_z , dim_y, dim_x), dtype=np.float32)
    radius32 = np.int32(radius)
    sigma32 = np.float32(sigma)
    kerGaussian = np.array([np.exp(-(float(R))**2/(2*sigma**2)) for R in range(-radius,radius+1)]).astype(np.float32)#(1./(np.sqrt(2*np.pi)*sigma))*
    #print kerGaussian
    print(dim_X,dim_Y,dim_Z)
    print(stackImages.shape)
    #print stackImages
    #((float(R)-radius)**2/(2*sigma**2))
    #Host to Device copy
    stack_gpu = to_gpu(stackImages)
    print("Stack_gpu copied")
    auxStack_gpu = to_gpu(auxStack)
    print("auxiliary Stack_gpu copied")
    kerGaussian_gpu = to_gpu(kerGaussian)
    print("gaussian kernel copied")
    print("Data transfered to GPU")

    print("Tokenizing filter")
    mod1 = SourceModule("""
    __global__ void d_gaussian_filter(float* stack_gpu, float* auxStack_gpu, float* cGaussian, int dim_x, int dim_y, int dim_z, int r)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int z = idx / (dim_x*dim_y);
        int x = (idx % (dim_x*dim_y) ) % dim_x;
        int y = (idx % (dim_x*dim_y) ) / dim_x; 

        if (x>=dim_x || y>=dim_y || z>=dim_z)
        {
            return;
        }

        float sum = 0.0;
        float factor=0.0;
        float t = 0.0;

        for (int i0 = -r; i0 <= r; i0++)
        {   int i=i0;
            if(x+i0>=dim_x || x+i0<0){i=-i0;}

            for (int j0 = -r; j0 <= r; j0++)
            {   int j=j0;
                if(y+j0>=dim_y || y+j0<0){j=-j0;}
                factor = cGaussian[r+i0] * cGaussian[r+j0];  
                t += factor * stack_gpu[z*dim_x*dim_y+(y+j)*dim_x+x+i];
                sum += factor;
            }
        }

        auxStack_gpu[idx] = t/sum;

    }
    """)
    print("Tokenizing copy_kernel")
    mod2 = SourceModule("""
    __global__ void copy_kernel(float* stack_gpu, float* auxStack_gpu)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        stack_gpu[idx] = auxStack_gpu[idx];
    }
    """)


    print("Defining kernel filter")
    func_gaussianFilter = mod1.get_function("d_gaussian_filter")
    func_copyKernel = mod2.get_function("copy_kernel")

    #Function calls
    print("Ready to calculate the filter")
    func_gaussianFilter(stack_gpu, auxStack_gpu, kerGaussian_gpu, dim_X, dim_Y, dim_Z, radius32, block=(block_X, block_Y, 1),
            grid=(grid_X, grid_Y))
    func_copyKernel(stack_gpu,auxStack_gpu, block=(block_X, block_Y, 1),
            grid=(grid_X, grid_Y))
    # func_gaussianFilter(stack_gpu, auxStack_gpu, kerGaussian_gpu, dim_X, dim_Y, dim_Z, radius32, block=(block_X, block_Y, 1),
    #         grid=(grid_X, grid_Y))
    # func_copyKernel(stack_gpu,auxStack_gpu, block=(block_X, block_Y, 1),
    #         grid=(grid_X, grid_Y))
    print("Done.")
   
    
    print("Copy to Host filtered images")
    stackImages = stack_gpu.get()
    print("Done")
    # As an alternative
    #driver.memcpy_dtoh(switch, switch_gpu)
    #driver.memcpy_dtoh(levels, levels_gpu)

    #Free GPU memory
    print("Clearing memory of GPU")
    stack_gpu.gpudata.free()
    auxStack_gpu.gpudata.free()
    kerGaussian_gpu.gpudata.free()
    #print stackImages_filtered
    return stackImages

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #stackImages = sys.argv[1]
    #radius = sys.argv[2]
    pic = int(sys.argv[1])
    regions = np.zeros((10,300,100)) 
    for i in range(10):
        regions[i,5*i:5+5*i,5*i:5+5*i]=1
    #regions[:,6:,6:] = 1
    sigma = 2
    im_filtered = gaussianFilter(regions, radius=10, sigma=sigma)
    im = [filters.gaussian(r, sigma=sigma) for r in regions]

    fig1, ax = plt.subplots(1,3,sharex=True, sharey=True)
    ax[0].imshow(regions[pic])
    ax[1].imshow(im[pic])
    ax[2].imshow(im_filtered[pic])
    plt.show()