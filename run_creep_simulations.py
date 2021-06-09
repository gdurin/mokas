import os
from run_creep import Creep 
import sys

def isIniFile(filename):
	return os.path.isfile(filename)

if __name__ == "__main__":
	myChoice = ['D0','D0.5'][1]
	if myChoice == 'D0':
		iniFilepath = "/home/wall/src/mumax3/bhaskar/bubble/finalSimulation/05062016/disorder/disorder_D0_sim.ini"
		# Select the OoP field
		Bz = -20
	elif myChoice == 'D0.5':
		iniFilepath = "/home/wall/src/mumax3/bhaskar/bubble/finalSimulation/05062016/disorder/disorder_D0.5_sim.ini"
		# Select the OoP field
		Bz = -10

	if not isIniFile(iniFilepath):
		print("There is a problem with the ini file {}: file not found".format(iniFilepath))
		sys.exit()

	creep_data = Creep(iniFilepath, Bz)
	creep_data.plot_results()