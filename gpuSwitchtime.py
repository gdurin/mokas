import numpy as np
import scipy.ndimage as nd
#import pycuda.autoinit
from pycuda.gpuarray import to_gpu
from pycuda.compiler import SourceModule
import mokas_gpu as mkGpu


def get_gpuSwitchTime(stackImages, convolSize=10, multiplier=1, 
    current_dev=None, ctx=None, block_size=(256,1), verbose=False):
    """
    Return a matrix with the positions of a step in a sequence for each pixel

    Parameters:
    ---------------
    """

    # =========================================
    # Set the card to work with: DONE EXTERNALLY
    # =========================================
    if verbose:
        print("working on card %s" % current_dev.name())
    used_device = ctx.get_device()
    # Convert to int32
    dim_z, dim_y, dim_x = stackImages.shape
    dim_Z, dim_Y, dim_X = np.int32(stackImages.shape)
    block_X, block_Y = block_size
    grid_X, grid_Y = dim_x*dim_y*dim_z / block_X if (dim_x*dim_y*dim_z % block_X)==0 else dim_x*dim_y*dim_z / block_X +1 , 1
    grid_X2, grid_Y2 = dim_x / block_X + 1, dim_y/ block_Y + 1
    grid_X = int(grid_X)
    grid_Y = int(grid_Y)
    grid_X2 = int(grid_X2)
    grid_Y2 = int(grid_Y2)
    if verbose:
        print("Print grid dimensions: ", grid_X, grid_Y)
    convolStack = np.zeros((dim_z , dim_y, dim_x), dtype=np.float32)
    switch = np.zeros((dim_y,dim_x), dtype=np.int32)
    levels = np.zeros((dim_y,dim_x), dtype=np.int32)
    switch_max = np.zeros((dim_y,dim_x), dtype=np.int32)
    levels_max = np.zeros((dim_y,dim_x), dtype=np.int32)
    convolSize32 = np.int32(convolSize)
    multiplier32 = np.int32(multiplier)
    #Host to Device copy
    stack_gpu = to_gpu(stackImages)
    if verbose:
        print("Stack_gpu copied")
    switch_gpu = to_gpu(switch)
    if verbose:
        print("Switch_gpu copied")
    levels_gpu = to_gpu(levels)
    if verbose:
        print("Level_gpu copied")
    switch_max_gpu = to_gpu(switch_max)
    if verbose:
        print("Switch_max_gpu copied")
    levels_max_gpu = to_gpu(levels_max)
    if verbose:
        print("Level_max_gpu copied")
    convolStack_gpu = to_gpu(convolStack)
    if verbose:
        print("convolStack_gpu copied")
        print("Data transfered to GPU")
        print("Tokenizing 1")
    # contracts the kernel size when approaching edges
    mod1_a = SourceModule("""
    __global__ void findconvolve1d(int *stack_gpu, float *convolStack_gpu, int dim_x, int dim_y, int dim_z, int convolSize,int multiplier0)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x; 
        if ( idx > dim_x*dim_y*dim_z)
            return;
        else{
            int k = idx/(dim_x*dim_y);
            int convolSizeReal = ( ( convolSize + k >= dim_z || -convolSize + k < 0 ) ? min(k,abs(dim_z-1-k)) : convolSize);
            float partialSum=0;     
            int multiplier=multiplier0; 
            for (int r0=-convolSizeReal; r0<convolSizeReal; r0++){
                int r=r0;
                if( r0 >= 0 ) 
                    multiplier=-multiplier0;
                partialSum+=(multiplier*stack_gpu[idx+r*dim_x*dim_y]);
                }
            convolStack_gpu[idx]=partialSum/convolSizeReal;
        }
    }
    """)
    # keeps constant value
    mod1_b = SourceModule("""
    __global__ void findconvolve1d(int *stack_gpu, float *convolStack_gpu, int dim_x, int dim_y, int dim_z, int convolSize,int multiplier0)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x; 
        if ( idx > dim_x*dim_y*dim_z)
            return;
        else{
            int k = idx/(dim_x*dim_y);
            //int i = idx%(dim_x*dim_y)/dim_y;
            //int j = idx % (dim_x*dim_y) % dim_y; 

            float partialSum=0;     
            int multiplier=multiplier0; 
            for (int r0=-convolSize; r0<convolSize; r0++){
                int r=r0;
                if( r0 >= 0 ) 
                    multiplier=-multiplier0;
                if( r0+k >= dim_z ) r=dim_z-1;
                if( r0+k < 0 )  r=0;
                partialSum+=(multiplier*stack_gpu[idx+r*dim_x*dim_y]);
                }
            convolStack_gpu[idx]=partialSum/convolSize;
        }
    }
    """)
    #Keeps constant at beginning and mirrors at end of sequence
    mod1_c = SourceModule("""
    __global__ void findconvolve1d(int *stack_gpu, float *convolStack_gpu, int dim_x, int dim_y, int dim_z, int convolSize,int multiplier0)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x; 
        if ( idx > dim_x*dim_y*dim_z)
            return;
        else{
            int k = idx/(dim_x*dim_y);
            float partialSum=0;     
            int multiplier=multiplier0; 
            int r;
            for (int r0=-convolSize; r0<convolSize; r0++){
                if( r0 >= 0 ) 
                    multiplier=-multiplier0;
                if( r0+k >= dim_z)
                    r=-r0;
                else if ( r0+k < 0 )
                    r=0;
                else r=r0;
                    
                partialSum+=(multiplier*stack_gpu[idx+r*dim_x*dim_y]);
                }
            convolStack_gpu[idx]=partialSum/convolSize;
        }
    }
    """)
    # mirrors values out of bound
    mod1 = SourceModule("""
	__global__ void findconvolve1d(int *stack_gpu, float *convolStack_gpu, int dim_x, int dim_y, int dim_z, int convolSize,int multiplier0)
	{
        int idx = threadIdx.x + blockIdx.x * blockDim.x; 
        if ( idx > dim_x*dim_y*dim_z)
            return;
        else{
            int k = idx/(dim_x*dim_y);
            //int i = idx%(dim_x*dim_y)/dim_y;
            //int j = idx % (dim_x*dim_y) % dim_y; 

            float partialSum=0;     
            int multiplier=multiplier0; 
            for (int r0=-convolSize; r0<convolSize; r0++){
                int r=r0;
                if( r0 >= 0 ) 
                    multiplier=-multiplier0;
                if( r0+k >= dim_z || r0+k < 0 )
                    r=-r0;
                partialSum+=(multiplier*stack_gpu[idx+r*dim_x*dim_y]);
                }
            convolStack_gpu[idx]=partialSum/convolSize;
        }
 	}
    """)

    if verbose:
        print("Tokenizing 2")

    mod2 = SourceModule("""
    __global__ void findmin(float *convolStack_gpu, int *switch_gpu, int *levels_gpu, int *switch_max_gpu, int *levels_max_gpu, int dim_x, int dim_y, int dim_z)
    {
    int len_kernel_half = 15;
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    int idy = threadIdx.y + blockIdx.y * blockDim.y; 
      if (idx >= dim_x || idy >= dim_y)
        return;
    int flat_id1 = idx + dim_x * idy ;
    int min=4294967295;
    int max=-4294967294;
    for(int idz = 0; idz <dim_z; idz++)
      {
        int flat_id = idx + dim_x * idy + (dim_x * dim_y) * idz;      
        if(convolStack_gpu[flat_id]<min)
        {
        min=convolStack_gpu[flat_id];
        switch_gpu[flat_id1]=idz;
        }
        if(convolStack_gpu[flat_id]>max)
        {
        max=convolStack_gpu[flat_id];
        switch_max_gpu[flat_id1]=idz;
        }
      }
        levels_gpu[flat_id1]=min;
        levels_max_gpu[flat_id1]=max;
    }
    """)
    # mod2_b = SourceModule("""
    # __global__ void findmin(float *convolStack_gpu, int *switch_gpu, int *levels_gpu, int dim_x, int dim_y, int dim_z)
    # {
    # int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    # if ( idx > dim_x*dim_y*dim_z)
    #         return;

    # else{
    #     int k = idx/(dim_x*dim_y);
    #     int i = idx%(dim_x*dim_y)/dim_y;
    #     int j = idx % (dim_x*dim_y) % dim_y; 

    #     if (k != 0)return;
        
    #     int min=4294967295;
    #     float mean,stdDev;
        
    #     for(int idz = 0; idz <dim_z; idz++)
    #     {
    #         if(convolStack_gpu[idx+idz*dim_x*dim_y]<min)
    #         {
    #         min=convolStack_gpu[idx+idz*dim_x*dim_y];
    #         switch_gpu[idx+idz*dim_x*dim_y]=idz;
    #         mean+=
    #         }
    #     }
    #     levels_gpu[flat_id1]=abs(min);
    # }
    # """)


    if verbose:
        print("Defining kernel convolve")
    func_findconvolve1d = mod1_c.get_function("findconvolve1d")
    # Get the array with the switching time
    if verbose:
        print("Defining kernel findmin")
    func_findmin = mod2.get_function("findmin")

    #Function calls
    if verbose:
        print("Ready to calculate the convolution")
    func_findconvolve1d(stack_gpu, convolStack_gpu, dim_X, dim_Y, dim_Z,convolSize32,multiplier32, block=(block_X, block_Y, 1),
         grid=(grid_X, grid_Y))
    if verbose:
        print("Done.")
        print("Ready to find the minimum of convolution")
    func_findmin(convolStack_gpu, switch_gpu, levels_gpu, switch_max_gpu, levels_max_gpu,  dim_X, dim_Y, dim_Z,  block=(block_X, block_Y, 1),
          grid=(grid_X2, grid_Y2))
    if verbose:
        print("Done")
        #Device to host copy  
        print("Copy to Host switchtimes")
    switch = switch_gpu.get()
    if verbose:
        print("Copy to Host levels")
    levels = levels_gpu.get()
    if verbose:
        print("Copy to Host switchtimes_max")
    switch_max = switch_max_gpu.get()
    if verbose:
        print("Copy to Host levels_max")
    levels_max = levels_max_gpu.get()
    if verbose:
        print("Done")
    # As an alternative
    #driver.memcpy_dtoh(switch, switch_gpu)
    #driver.memcpy_dtoh(levels, levels_gpu)

    #Free GPU memory
    if verbose:
        print("Clearing memory of GPU")
    stack_gpu.gpudata.free()
    switch_gpu.gpudata.free()
    switch_max_gpu.gpudata.free()
    convolStack_gpu.gpudata.free()
    levels_gpu.gpudata.free()
    levels_max_gpu.gpudata.free()

    return switch, levels, switch_max, levels_max


if __name__ == "__main__":
    import time
    import mokas_gpu as gpu
    current_dev, ctx, (free, total) = gpu.gpu_init(1)
    # Prepare a 3D array of random data as int32
    dim_x = 150
    dim_y = 150 
    dim_z = 80
    a = np.random.randn(dim_z,dim_y,dim_x)
    a = a.astype(np.int32)
    print("Loading %.2f MB of data" % (a.nbytes/1e6))
    # Call the GPU kernel
    kernel = np.array([-1]*15+[1]*15)
    t0 = time.time()
    gpuswitch, gpulevels = get_gpuSwitchTime(a, kernel, current_dev=current_dev, ctx=ctx, block_size=(64,4))	
    timeGpu = time.time() - t0
    # Make the same calculation on the CPU
    step = kernel
    cpuswitch=np.zeros((dim_y,dim_x),dtype=np.int32)
    cpulevels=np.zeros((dim_y,dim_x),dtype=np.int32)
    print("Loading %.2f MB of data" % (2*cpuswitch.nbytes/1e6))
    t3=time.time()
    for i in range(0,dim_x):
        for j in range(0,dim_y):
            indice=(nd.convolve1d(a[:,j,i],step,mode='reflect')).argmin()      
            cpuswitch[j,i]=indice
    timeCpu = time.time()-t3
    print("GPU calculus done = %.4f s" %timeGpu)
    print("CPU calculus done = %.4f s" %timeCpu)
    print("Difference on switch : \n")
    print(gpuswitch-cpuswitch)
    print("\nGPU is %d times faster than CPU " %(timeCpu/timeGpu))
    gpu.gpu_deinit(current_dev, ctx)