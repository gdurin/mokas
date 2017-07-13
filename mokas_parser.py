import os, sys
import numpy as np
from collections import OrderedDict, defaultdict
import h5py


class Parser:
    """
    Class to deal with specific files used in experiments
    measure.txt : property of the instrument
    """
    def __init__(self, filename):
        self.filename = filename
        mainDir, fname = os.path.split(filename)
        if fname == 'measure.txt':
            self.structure = {'camera': ['camera', 'Imager'],
                          'lens' : ['Epiplan', 'Plan', 'Melles'],
                          'pinhole' : ['pinhole', 'slit'],
                          'polarizer' : ['polarizer'],
                          'illumination' : ['illumination', 'fiber'], 
                          'others' : ['coil', 'electromagnet', 'generator', 'supply', 'Avtech', 
                          'Highland', 'amplifier']
                          }
            self.cameras = ['Zeiss', 'ThorLabs' , 'PicoStar']
            self.cameras_sizes = [None, '1392x1040', None]
            self.data_type = 'measure'
        else:
            print("Not installed yet!")
            sys.exit()

    def get_data(self):
        """
        get the dictionary of the data in the file
        """
        if self.data_type == 'measure':
            return self.parse_measure()

    def parse_measure(self):
        #data = defaultdict(list)
        data = {}
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        lines0 = [l[1:].strip() for l in lines if "[X]" in l or "\n"==l]
        lines0 = [l.replace("[X] ", "").strip() for l in lines0]
        for key, elements in self.structure.iteritems():
            data[key] = [l for l in lines0 for el in elements if el in l]
        # This is pretty brutal, but better than anything
        lens = data['lens'][0]
        lens = lens.split()[-3:]
        index = [i for i,c in enumerate(self.cameras) if c in data['camera'][0]][0]
        px = lens[index]
        # TODO Zeiss has a problem
        if "x" in px:
            x,y = [np.float(p) for p in px.split("x")]
            pixels_x, pixels_y = [np.int(p) for p in self.cameras_sizes[index].split("x")]
            um_per_pixel = (x/pixels_x + y/pixels_y) / 2.
            #print("%f um/pixel" % um_per_pixel)
        else:
            um_per_pixel = px
        data['um_per_pixel'] = um_per_pixel
        return data

if __name__ == "__main__":
    filename = '/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_new/20um/20um_0.11A/measure.txt'
    p = Parser(filename)
    data = p.get_data()

