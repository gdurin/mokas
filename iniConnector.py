# Connector to the ini file using the factory pattern
# The idea is to create a connector 
# for experimental and simulation data
# which is transparent and does not require any user input
# except for the ini file

import os,sys
import configparser
import colorsys
import numpy as np

def from_string(s, s_type='float'):
	"""
	get the float from a string,
	getting rid of comments
	out:
		float
		list_of_string
		list_of_float
	"""
	out_s = s.split("#")[0]
	out_s = out_s.strip()
	if s_type == 'float':
		out_s = np.float(out_s)
	if s_type == 'list_of_string' or s_type == 'list_of_float':
		out_s = out_s.strip("[]").split(",")
	if s_type == 'list_of_float':
		out_s = [np.float(element) for element in out_s]
	if s_type == 'list_of_tuple':
		out_s = eval(out_s)
	if s_type == 'tuple':
		out_s = out_s.strip("()")
		out_s = tuple([int(c) for c in out_s.split(",")])
	if s_type == 'int':
		out_s = np.int(out_s)
	return out_s

def to_list_of_string(s):
	return from_string(s,'list_of_string')

def to_list_of_float(s):
	return from_string(s,'list_of_float')

def to_float(s):
	return from_string(s,'float')

def to_int(s):
	return from_string(s,'int')

def to_string(s):
	return from_string(s,None)

def to_tuple(s):
	return from_string(s,'tuple')

def to_list_of_tuple(s):
	return from_string(s,'list_of_tuple')

class IniConnector:
	def __init__(self, filepath, Bz):
		self.config = configparser.ConfigParser()
		self.config.read(filepath)
		self.default = self.config['DEFAULT']
		self.__parse_iniFile(Bz)

	def __parse_iniFile(self,Bz):
		self.imageParameters = {}
		# General parameters
		self.material = to_string(self.default['material'])
		self.material_full = to_string(self.default['material_full'])
		self.frame_rate = to_float(self.default['frame_rate'])
		self.microns_per_pixel = to_float(self.default['microns_per_pixel'])
		self.imageParameters['pattern'] = to_string(self.default['pattern'])
		self.gray_threshold = to_float(self.default['gray_threshold'])

		if self.default['resize_factor'] == 'None':
		 	self.imageParameters['resize_factor'] = None
		else:
		 	self.imageParameters['resize_factor'] = to_float(self.default['resize_factor'])
		self.imageParameters['filtering'] = to_string(self.default['filtering'])
		self.imageParameters['sigma'] = to_float(self.default['sigma'])
		self.imageParameters['kernel_half_width_of_ones'] = to_int(self.default['kernel_half_width_of_ones'])

		# Read the available OoP and InPlane fields
		self.Bz_s_labels = to_list_of_string(self.default['Bz_s'])
		#print(self.default['Bz_s'])
		#print(self.Bz_s_labels)
		self.Bz_s = [np.float(b) for b in self.Bz_s_labels]
		self.Bx_s_labels = to_list_of_string(self.default['Bx_s'])
		self.Bx_s = [np.float(b) for b in self.Bx_s_labels]
		self.Bz_unit = self.default['Bz_unit']
		self.Bx_unit = self.default['Bx_unit']
		
		# Check if the Bz choosen is in the iniFile
		# and save it as a label (string)
		try:
			index = self.Bz_s.index(Bz)
			self.Bz_label = self.Bz_s_labels[index]
			self.Bz = Bz
		except:
			raise ValueError('Bz value %s not present' % Bz)
			sys.exit()

		# step in the frames to show a contours line in bold
		# Assumed fixed for each Bz
		crop = to_list_of_tuple(self.config[self.Bz_label]['imCrop'])
		self.imageParameters['imCrop'] = crop
		self.step_in_frames = to_int(self.config[self.Bz_label]['step_in_frames'])
		self.varsBx = {}
		for i,Bx in enumerate(self.Bx_s):
			label = "%s %s" % (self.Bz_label, self.Bx_s_labels[i])
			section = self.config[label]
			self.varsBx[Bx] = {}
			self.varsBx[Bx]['firstIm'] = to_int(section['firstIm'])
			self.varsBx[Bx]['lastIm'] = to_int(section['lastIm'])
			self.varsBx[Bx]['subDir'] = section['subDir']

class ExperimentalConnector(IniConnector):
	def __init__(self, filepath, Bz):
		IniConnector.__init__(self, filepath, Bz)
		for i,Bx in enumerate(self.Bx_s):
			label = "%s %s" % (self.Bz_label, self.Bx_s_labels[i])
			section = self.config[label]
			self.varsBx[Bx]['pulse_duration'] = to_float(section['pulse_duration'])

	@property
	def Bz_mT(self):
		if self.Bz_unit == 'mT':
			return self.Bz
		elif self.Bz_unit == 'Oe':
			return 0.1 * self.Bz
		elif self.Bz_unit == 'V':
			return self.Bz * to_float(self.default['Bz_V_to_mT'])
		

class SimulationConnector(IniConnector):
	def __init__(self, filepath, Bz):
		IniConnector.__init__(self, filepath, Bz)	
		for i,Bx in enumerate(self.Bx_s):
			self.varsBx[Bx]['pulse_duration'] = None

	@property
	def Bz_mT(self):
		if self.Bz_unit == 'mT':
			return self.Bz
		elif self.Bz_unit == 'Oe':
			return 0.1 * self.Bz


def connection_factory(filepath, Bz):
	basename = os.path.basename(filepath)
	if "_sim.ini" in basename:
		connector = SimulationConnector
	elif "_exp.ini" in basename:
		connector = ExperimentalConnector
	else:
		raise ValueError('File not found, or the ini file does not end with _sim.ini or _exp.ini')
	return connector(filepath, Bz)

def connect_to(filepath, Bz):
	"""
	Connect to one of the classes (Connectors)
	for experimental or simulation data
	"""
	factory = None
	try:
		factory = connection_factory(filepath, Bz)
	except ValueError as ve:
		print(ve)
	return factory