#import pycuda.autoinit
import pycuda.driver as driver
import sys

def gpu_init(device=0, fraction_to_use=0.95, verbose=False):
    try:
        # =========================================
        # Set the card to work with
        driver.init()
        current_dev = driver.Device(device)
        ctx = current_dev.make_context()
        ctx.push()
        free, total = driver.mem_get_info()
        # =========================================
        # print("current device: %s" % current_dev.name())
        free = fraction_to_use * free
        if verbose:
            print("Total memory of %s: %.2f GB (available: %.2f GB)" % (current_dev.name(), total/1e9, free/1e9))
    except:
        print("pyCUDA not installed properly or device %i not available" % device)
        sys.exit()
    return current_dev, ctx, (free, total)

def gpu_deinit(current_dev, ctx):
    name = current_dev.name()
    try:
        ctx.pop()
        ctx.detach()
    except:
        print("There problems to close device %s" % name)
        return False
    finally:
        print("Device %s closed properly" % name)
    return True