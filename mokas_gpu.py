#import pycuda.autoinit
import pycuda.driver as driver

def gpu_init(device=0):
    try:
        # =========================================
        # Set the card to work with
        driver.init()
        current_dev = driver.Device(device)
        ctx = current_dev.make_context()
        ctx.push()
        free, total = driver.mem_get_info()
        # =========================================
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