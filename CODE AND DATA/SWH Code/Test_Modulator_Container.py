import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt

import glob
import re
import pathlib

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")
sys.path.append("holotorch_addons/")

# import holotorch.utils.Dimensions as Dimensions
# from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
from holotorch.HolographicComponents.SLM_PhaseOnly import SLM_PhaseOnly
from holotorch.utils.Enumerators import *

import holotorch_addons.HolotorchPatches as HolotorchPatches
from holotorch_addons.HelperFunctions import check_tensors_broadcastable


############################################################################
# NOTES:
# 	- The 2020 WISHED paper used a HOLOEYE LETO SLM for the experiments
#		- The resolution of the SLM was 1920x1080 pixels (Width x Height)
# 		- Presumably, the LETO-3-NIR-081 was used as that is the only model
#		  that operated at the wavelengths used in the experiment (~854 nm)
#		- The maximum phase shifts were specified as:
#			- 4.4*pi at 650 nm, and 2.4*pi at 1064 nm
#				- Doing linear interpolation between those points gives
#				  a maximum phase shift of 3.414492756*pi at 854 nm
#					- The 3.414492756*pi value can be used for the
#					  'init_variance' parameter in the phase-only SLM component
############################################################################
slmMaxPhaseShift = 3.414492756 * np.pi

data_folder = pathlib.Path("../DATA")
temp_slm_data_folder = data_folder / ".temp_slm"
print(data_folder.resolve())
print(temp_slm_data_folder.resolve())

# slm_data_folder = data_folder / "slm_patterns"
# print(slm_data_folder.resolve())
# # slm.save_all_slms_into_folder(slm_data_folder)
# # slm.load_all_slms_from_folder(slm_data_folder)

def verifyNoAccidentalDuplication(slmTest):
	for i in range(len(slmTest)):
		for j in range(i+1, len(slmTest)):
			if (torch.equal(slmTest[i]['_data_tensor'], slmTest[j]['_data_tensor'])): 	# EXTREMELY EXTREMELY small chance that this could be true when nothing was wrong.  Would require that two randomly generated tensors have the exact same values though.
				return False
	return True


def checkForMatchingData(slm, slmTest, n_load_tests):
	for i in torch.randint(len(slmTest), (1,n_load_tests))[0].numpy():
		tempVals = slm.load_single_slm(batch_idx=i)
		if not (torch.equal(slmTest[i]['_data_tensor'], tempVals.data_tensor)):
			return False
	return True

def loadSlmDataFolder(folder):
	files = glob.glob(str(folder)+"\\*.pt")
	slmData = [None] * len(files)
	for i in range(len(files)):
		slmData[i] = torch.load(pathlib.Path(files[i]).resolve())
	return slmData

n_slm_batches = np.random.randint(1,6)
n_batch = np.random.randint(1,6) * n_slm_batches
slm =	HolotorchPatches.SLM_PhaseOnly_Patched.create_slm(	height          = 270,
															width           = 480,
															n_channel       = np.random.randint(1,4),
															n_batch			= n_batch,
															n_slm_batches	= n_slm_batches,
															feature_size    = 6.4*um,
															init_type       = ENUM_SLM_INIT.RANDOM,
															init_variance   = slmMaxPhaseShift,
															slm_directory	= temp_slm_data_folder.resolve(),
															store_on_gpu	= True
														)

slm.save_all_slms_into_folder(data_folder / 'Testing' / 'test3')
slmTest3 = loadSlmDataFolder(data_folder / 'Testing' / 'test3')
print("Test pass status:", checkForMatchingData(slm, slmTest3, np.random.randint(50,101)))
print("Test pass status:", verifyNoAccidentalDuplication(slmTest3))

rand_idx = np.random.randint(0,n_slm_batches)
slm.save_single_slm(batch_idx = rand_idx, folder = data_folder / 'Testing' / 'test_blah', filename='blah1234')

slmTest1 = loadSlmDataFolder(data_folder / "Testing" / "test1")
slm.load_all_slms_from_folder(data_folder / "Testing" / "test1")
print("Test pass status:", checkForMatchingData(slm, slmTest1, np.random.randint(50,101)))
print("Test pass status:", verifyNoAccidentalDuplication(slmTest1))

slmTest2 = loadSlmDataFolder(data_folder / "Testing" / "test2")
slm.load_all_slms_from_folder(data_folder / "Testing" / "test2")
print("Test pass status:", checkForMatchingData(slm, slmTest2, np.random.randint(50,101)))
print("Test pass status:", verifyNoAccidentalDuplication(slmTest2))

slm.load_all_slms_from_folder(data_folder / "Testing" / "test3")
print("Test pass status:", checkForMatchingData(slm, slmTest3, np.random.randint(50,101)))

slm.load_all_slms_from_folder(data_folder / "Testing" / "test2")
print("Test pass status:", checkForMatchingData(slm, slmTest2, np.random.randint(50,101)))

tempAsdf = torch.load(data_folder / 'Testing' / 'test_blah' / 'blah1234')
print("Test pass status:", torch.equal(tempAsdf['_data_tensor'], slmTest3[rand_idx]['_data_tensor']))

slm.save_all_slms_into_folder(data_folder / 'Testing' / 'test5')
slmTest5 = loadSlmDataFolder(data_folder / 'Testing' / 'test5')
print("Test pass status:", checkForMatchingData(slm, slmTest5, np.random.randint(50,101)))

slm.load_all_slms_from_folder(data_folder / "Testing" / "test1")
print("Test pass status:", checkForMatchingData(slm, slmTest1, np.random.randint(50,101)))

slm.load_all_slms_from_folder(data_folder / "Testing" / "test5")
print("Test pass status:", checkForMatchingData(slm, slmTest5, np.random.randint(50,101)))

slm.load_all_slms_from_folder(data_folder / "Testing" / "test1")
print("Test pass status:", checkForMatchingData(slm, slmTest1, np.random.randint(50,101)))

slm.load_all_slms_from_folder(data_folder / "Testing" / "test2")
print("Test pass status:", checkForMatchingData(slm, slmTest2, np.random.randint(50,101)))

pass