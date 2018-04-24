import os
import numpy as np
import h5py
from collections import OrderedDict
from pathlib import Path

dir_logic = {}

#################### Wires ###############################
# Set the logic of the root_dir of the LAST n items
# example:
# /home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_old/20um/20um_0.145A/20um_0.145A_10fps_2
dir_logic['Arianna'] = {}
dir_logic['Arianna']['from_root_dir'] = ['material', 'width', 'width_H', 'baseName']
# baseDir = /home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_old
# width = 20um
# width_H = 20um_0.145A
# baseName = 20um_0.145A_10fps_2
dir_logic['Arianna']['hdf5_root_dir'] = 'material'
# Give the logic for the full pattern using "_" as separator
dir_logic['Arianna']['from_pattern'] = ['width', 'H', 'fps', 'n_exp', 'MMS', 'Pos0']
dir_logic['Arianna']['hdf5_dirs'] = ['width', 'H', 'fps', 'n_wire', 'n_exp']
# 20um_0.145A_10fps_2_MMStack_Pos0.ome.tif
# nexp = 2
# fps = 10fps
#################### Bubbles ###############################
# Set the logic of the root_dir of the LAST n items
# example:
# /data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/Dec2016/0.116A/01_Irr_800uC_0.116A
dir_logic['SuperSlowCreep'] = {}
dir_logic['SuperSlowCreep']['from_root_dir'] = ['year', 'H', 'baseName']
# baseDir = /data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/Dec2016
# H = 0.116A
# baseName = 01_Irr_800uC_0.116A
dir_logic['SuperSlowCreep']['hdf5_root_dir'] = 'year'
# Give the logic for the full pattern using "_" as separator

#dir_logic['SuperSlowCreep']['from_pattern'] = ['n_exp', 'mat1', 'mat2', 'H', 'MMS', 'Pos0']
dir_logic['SuperSlowCreep']['from_pattern'] = ['n_exp', 'mat1', 'H', 'MMS', 'Pos0']   #for NonIrr

dir_logic['SuperSlowCreep']['hdf5_dirs'] = ['H', 'n_exp']
dir_logic['SuperSlowCreep']['hdf5_root_dir'] = 'year'
dir_logic['SuperSlowCreep']['hdf5_baseName'] = 'year'
# 01_Irr_800uC_0.116A_MMStack_Pos0.ome.tif
# H = 0.116A

################ With Set #############################################
dir_logic['SuperSlowCreep_withSet'] = {}
dir_logic['SuperSlowCreep_withSet']['from_root_dir'] = ['year', 'H', 'set_n', 'baseName']
# baseDir = '/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/NonIrr/Feb2018/0.137A/Set1/03_NonIrr_0.137A/'
# H = 0.137A
# set_n = Set1
# baseName = Set1/03_NonIrr_0.137A
# Give the logic for the full pattern using "_" as separator

#dir_logic['SuperSlowCreep']['from_pattern'] = ['n_exp', 'mat1', 'mat2', 'H', 'MMS', 'Pos0']
dir_logic['SuperSlowCreep_withSet']['from_pattern'] = ['n_exp', 'mat1', 'H', 'MMS', 'Pos0']   #for NonIrr
dir_logic['SuperSlowCreep_withSet']['hdf5_dirs'] = ['set_n', 'H', 'n_exp']
dir_logic['SuperSlowCreep_withSet']['hdf5_root_dir'] = 'H'
dir_logic['SuperSlowCreep_withSet']['hdf5_baseName'] = 'H'

class RootHdf5:
    """
    class to handle the main hdf5 in the root directory
    """
    def __init__(self, root_dir, pattern, signature, verbose=False):
        params = {}
        d = {}
        if verbose:
            print(50*"#")

        user = signature['user']
        # Load a logic for the dirs as defined above
        self._dl = dir_logic[user]
        # Calculate how many items has to be considered in the root_dir
        # reverse the direction of the lists
        in_root = self._dl['from_root_dir']
        len_in_root = len(in_root)
        pt = Path(root_dir).parts[-len_in_root:]
        for key, item in zip(in_root, pt):
            params[key] = item
        # Get the elements from the patterm
        # First the keys in the pattern (filename)
        in_pattern = self._dl['from_pattern']
        if verbose:
            print("in_pattern: %s" % in_pattern)
        pt = pattern.split("_")
        for key, item in zip(in_pattern, pt):
            if key not in params:
                params[key] = item
        # Get the elements for the baseGroup
        self.baseGroup = []
        for key in self._dl['hdf5_dirs']:
            print(params[key])
            self.baseGroup.append(params[key])
        self.baseGroup = os.path.join(*self.baseGroup)

        _from_root_dir = self._dl['from_root_dir']
        _len_from_root_dir = len(_from_root_dir)
        _position = _from_root_dir.index(self._dl['hdf5_root_dir'])
        k = _len_from_root_dir - 2 - _position
        baseDir = Path(root_dir).parents[k]
        baseDir = str(baseDir)
        if verbose: 
            print("baseDir: %s" % baseDir)

        _position = _from_root_dir.index(self._dl['hdf5_baseName'])
        k = _len_from_root_dir - _position
        baseName = Path(root_dir).parts[-k]
        hdf5_filename = str(baseName) + ".hdf5"
        if verbose: 
            print("hdf5_filename: %s" % hdf5_filename)

        signature_hdf5 = self.to_signature_hdf5(signature)
        if verbose: print(signature_hdf5)
        #dirs.update(signature_hdf5)
        
        self.fname = os.path.join(baseDir, hdf5_filename)
        is_hdf5 = os.path.isfile(self.fname)

        if verbose:
            print("baseGroup:")
            print(self.baseGroup)
        
        signature_hdf5 = self.to_signature_hdf5(signature)
        with h5py.File(self.fname, 'a') as f:
            dt = h5py.special_dtype(vlen=str)
            if self.baseGroup not in f:
                grp0 = f.create_group(self.baseGroup)
                print("Group %s created" % self.baseGroup)
                for key, item in signature_hdf5.iteritems():
                    grp0.attrs.create(key, item, dtype=dt)
            else:
                grp0 = f[self.baseGroup]
                print("Group %s exists" % self.baseGroup)
                print("Updating the signature")
                grp0.attrs.update(**signature_hdf5)
            # TODO: check the signature too
            self.is_raw_images = 'images' in grp0
        
    def load_raw_images(self):
        with h5py.File(self.fname, 'r') as f:
            grp0 = f[self.baseGroup]
            if self.is_raw_images:
                images = grp0['images'][...]
                imageNumbers = grp0['imageNumbers'][...]
            else:
                print("No data available, please check")
                images, imageNumbers = None, None
        return images, imageNumbers

    def load_obj(self, obj):
        with h5py.File(self.fname, 'r') as f:
            grp0 = f[self.baseGroup]
            if obj in grp0:
                return grp0[obj][...]
            else:
                print("No %s data available, please check" % obj)
                return None

    def load_dict(self, label, key_type=np.int):
        """
        load a dictionary written as groups
        under the 'label' group
        Parameters:
            label : str
                group where the dict is saved
            key_type : number type
                transform the key into a number, if possible
        """
        d = {}
        with h5py.File(self.fname, 'r') as f:
            grp0 = f[self.baseGroup][label]
            for group in grp0: #yes, it looks like a dictionary
                if key_type:
                    key = key_type(group)
                else:
                    key = group
                d[key] = grp0[group][...]
        return d

    def save_data(self, datas, labels, dtype):
        """
        Save the data passed as a list of data and labels
        """
        if isinstance(datas,dict):
            self.save_dict(datas, labels, dtype)
        else:
            if not isinstance(datas, list):
                datas = [datas]
                labels = [labels]
            with h5py.File(self.fname, 'a') as f:
                grp0 = f[self.baseGroup]
                for data, label in zip(datas, labels):
                    if not label in grp0:
                        dset = grp0.create_dataset(label, data=data, dtype=dtype)
                    else:                    
                        grp0[label][...] = data
                f.flush()
        return True

    def save_raw_images(self, images, imageNumbers, dtype=np.int16):
        """
        Save the images as numpy 3D arrays
        with the imageNumbers array
        """
        datas = [images, imageNumbers]
        labels = ['images', 'imageNumbers']
        success = self.save_data(datas, labels, dtype=dtype)
        return success


    def save_cluster2D(self, cluster2D, dtype=np.int16):
        success = self.save_data(cluster2D, 'cluster2D', dtype=dtype)       
        return success

    def save_dict(self, dic, label, dtype):
        """
        recursively save the content of a dict to group
        https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
        """
        with h5py.File(self.fname, 'a') as f:
            grp0 = f[self.baseGroup]
            if label in grp0:
                grp0 = grp0[label]
            else:
                grp0 = grp0.create_group(label)
            self._recursively_save_dict_contents_to_group(grp0, dic, dtype)
            
    def _recursively_save_dict_contents_to_group(self, path, dic, dtype):
        """
        there is a potential problem here:
        it the keys of a dictonary is a number (int, float)
        it has to be transformed into a str to create the group
        """
        for key, item in dic.items():
            # Force to be a string
            key = str(key)
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                if not key in path:
                    dset = path.create_dataset(key, data=item, dtype=dtype)
                else:
                    # Problems if the item exists and has a different shape
                    # example:  TypeError: Can't broadcast (1269, 2) -> (7, 2)
                    dset = path[key]
                    try:
                        dset[...] = item
                    except TypeError:
                        del path[key] # Note that del dset does not work as expected
                        path.create_dataset(key, data=item, dtype=dtype)
            elif isinstance(item, dict):
                self._recursively_save_dict_contents_to_group(path[key], item)
            else:   
                raise ValueError('Cannot save %s type'%type(item))


    def to_signature_hdf5(self, signature_np):
        """
        check if the dictionary of the signature
        contains bool or None
        and change them into strings
        """
        signature_hdf5 = {}
        for key, elem in signature_np.iteritems():
            signature_hdf5[key] = np.string_(elem)
        return signature_hdf5

    def to_signature_numpy(self, signature_hdf5):
        signature_np = np.copy(signature_hdf5)
        for key, elem in signature_hdf5.iteritems():
            if elem is 'True' or elem is 'False':
                signature_np[key] == 'True'
            elif elem is 'None':
                signature_np[key] = None
        return signature_np



    
if __name__ == "__main__":
    import scipy.misc
    root_dir = "/home/gf/Meas/Creep/CoFeB/Wires/Arianna/Ta_CoFeB_MgO_wires_IEF_old/20um/20um_0.145A/20um_0.145A_10fps_2"
    pattern = "20um_0.145A_10fps_2_MMStack_Pos0.ome.tif"
    # root_dir = "/home/gf/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/01_Irr_800uC_0.116A"
    # pattern = "01_Irr_800uC_0.116A_MMStack_Pos0.ome.tif"
    
    signature = {'firstIm':0, 'lastIm':-1, 'crop':None, 
                'rotation':None, 'filtering':'gauss', 'sigma':1, 'user':'Arianna', 'n_wire':'wire1'}
    # rd = RootHdf5(root_dir, pattern, signature)
    # # 
    # image = scipy.misc.ascent()
    # image = image[np.newaxis,...]
    # images = np.vstack((image, image))
    # imageNumbers = range(1,3,1)
    # success = rd.save_raw_images(2*images, imageNumbers)
    # success = rd.save_cluster2D(images)
    # Load a dictionary
    #root_dir = "/data/Meas/Creep/CoFeB/Film/SuperSlowCreep/Irr_800uC/Dec2016/Irr_800uC.hdf5"
    rd = RootHdf5(root_dir, pattern, signature)