import os, glob
import configparser
import matplotlib.pyplot as plt
from visualBarkh import StackImages
import mahotas
import skimage.morphology as morph
import skimage.feature as feature

class Wires_ini(object):
    def __init__(self, filepath, n_wire=1):
        self.imParameters = dict()
        self.config = configparser.ConfigParser()
        filename = os.path.join(filepath, "wires.ini")
        if not os.path.isfile(filename):
            print("Please, prepare a wires.ini file")
        self.config.read(filename)
        self.default = self.config['DEFAULT']
        self.n_wires = int(self.default['n_wires'])
        self.filename_suffix = self.default['filename_suffix']
        self.imParameters['firstIm'] = int(self.default['firstIm'])
        self.imParameters['lastIm'] = int(self.default['lastIm'])
        self.imParameters['filtering'] = self.default['filtering']
        self.imParameters['sigma'] = float(self.default['sigma'])
        self.imParameters['rotation'] = float(self.default['rotation'])
        if n_wire > self.n_wires:
            print("Number of the wire not available (1-%i)" % self.n_wires)
        nwire = "n%i" % n_wire
        nw = self.config[nwire]
        crop_upper_left_pixel = tuple([int(n) for n in nw['crop_upper_left_pixel'].split(",")])
        crop_lower_right_pixel = tuple([int(n) for n in nw['crop_lower_right_pixel'].split(",")])
        self.imParameters['imCrop'] = [crop_upper_left_pixel, crop_lower_right_pixel]
        self.experiments = [int(n) for n in nw['experiments'].split(",")]


class Wires(StackImages):
    """
    define a proper class to handle
    the sequence of images
    taken from wires
    """
    def __init__(self, **imParameters):
        StackImages.__init__(self, **imParameters)

    def stats_prop(self, min_size=30):
        """
        calculate the statistical properties 
        of the avalanches
        """
        q = np.copy(self._switchTimes2D)
        self.switches = np.unique(q)[1:]
        # Calculation of sizes
        self.sizes_whole = np.array([sum(q == sw) for sw in self.switches])
        # Here we need to get the largest cluster and get its properties
        sizes = []
        for i, sw in enumerate(self.switches):
            im = q == sw
            largest_cluster, cluster_size = self._largest_cluster(im)
            if cluster_size < min_size:
                self.switches = np.delete(self.switches, i)
            else:
                sizes.append(cluster_size)


    def _lenghts(self, cluster):
        cluster = morph.remove_small_holes(cluster)
        medial_axis = morph.medial_axis(cluster)
        # Find the corners with the corner_fast method
        cf = feature.corner_fast(cluster)
        # Select the two corners (little clusters) 
        # farthest from the center of mass

    

    def _largest_cluster(self, im):
        """
        find the largest cluster in a image
        """
        im, n_clusters = mahotas.label(im)
        if n_clusters = 1:
            return im
        else:
            sizes = mahotas.labeled.labeled_size(im)[1:]
            i = np.argmax(sizes)
            return im==i+1, sizes[i]

    def find_contours(self, lines_color=None, invert_y_axis=True, step_image=1,
                        consider_events_around_a_central_domain=True, 
                        initial_domain_region=None, remove_bordering=False,
                        plot_centers_of_mass = False, reference=None, 
                        rescale_area=False, plot_rays=True,
                        fig=None, ax=None, title=None):
        if fig is None:
            fig = plt.figure(figsize=self._figColorImage.get_size_inches())
            ax = fig.gca()
        else:
            plt.figure(fig.number)
            if ax is None:
                ax = fig.gca()
        self.contours = {}
        print("Sorry, not yet implemented")