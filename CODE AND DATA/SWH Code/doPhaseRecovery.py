from logging import exception
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

import pathlib
import glob
import copy
import datetime

from numpy import asarray
import gc	# For garbage collection/freeing up memory

# Image wranglers
import imageio
from PIL import Image

import warnings

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")
sys.path.append("holotorch_addons/")

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
# from holotorch.LightSources.CoherentSource import CoherentSource
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
from holotorch.utils.Enumerators import *
# from holotorch.Optical_Components.Resize_Field import Resize_Field
from holotorch.Sensors.Detector import Detector
from holotorch.CGH_Datatypes.IntensityField import IntensityField

from holotorch_addons.HelperFunctions import computeSpatialFrequencyGrids, computeBandlimitASM, computeBandlimitingFilterSpaceDomain, computeBandlimitingFilterASM, get_field_slice, print_cuda_memory_usage
from holotorch_addons.Save_Load_Functions import findNoncollidingFilepath, infer_parameters_from_save_data, recreate_slm_to_sensor_plane_model_for_v1_data, loadSensorData
from holotorch_addons.Field_Resampler import Field_Resampler
from holotorch_addons.SimpleDetector import SimpleDetector
# from holotorch_addons.Thin_Lens import Thin_Lens
from ASM_Prop_Patched import ASM_Prop_Patched

import holotorch_addons.HolotorchPatches as HolotorchPatches
from Synthetic_Data_Generator import Synthetic_Data_Generator

################################################################################################################################




################################################################################################################################



################################################################################################################################




################################################################################################################################
# dataFolderPathStr = '../DATA/Synthetic Data/DATA_2022-9-10_21h07m38s'
# dataFolderPathStr = '../DATA/Synthetic Data/DATA_2022-9-11_16h18m11s'
# dataFolderPathStr = '../DATA/Synthetic Data/DATA_2022-9-13_02h43m19s'

# dataFolderPathStr = '../DATA/Synthetic Data/DATA_2022-9-18_07h06m11s'

dataFolderPathStr = '../DATA/Synthetic Data/DATA_2022-10-2_20h22m28s'
temp_slm_data_folder_str = "../DATA/.temp_slm"

use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
storeSlmTempDataOnGpu = True
################################################################################################################################


dataFolderPath = pathlib.Path(dataFolderPathStr)
loadedInfo = infer_parameters_from_save_data(dataFolderPath)
slmInputToSensorOutputModel, slmModel, slmToSensorPlaneModel, detectorModel = recreate_slm_to_sensor_plane_model_for_v1_data(loadedInfo, dataFolderPath, pathlib.Path(temp_slm_data_folder_str), storeSlmTempDataOnGpu=storeSlmTempDataOnGpu, device=device)
sensorData = loadSensorData(dataFolderPath=dataFolderPath, device=device)


slmPlaneResolution = loadedInfo['slmPlaneResolution']
slmPlanePixelPitch = loadedInfo['slmPlanePixelPitch']
n_slm_batches = loadedInfo['n_slm_batches']		# Number of mini-batches


# Optimization settings
maxIterations = 500
plateauDeltaThreshold = 1e-7
plateauIterationsThreshold = 20


wavelengths = copy.deepcopy(loadedInfo['wavelengths'])
inputFieldSize = torch.Size([1, 1, 1, wavelengths.tensor_dimension.channel, slmPlaneResolution[0], slmPlaneResolution[1]])
initData = torch.tensor(1*torch.ones(inputFieldSize, dtype=torch.cfloat, device=device), requires_grad=True)
inputField = 	ElectricField(
					data = initData,
					wavelengths = wavelengths,
					spacing = slmPlanePixelPitch
				)
inputField.spacing = inputField.spacing.to(device)

optimizer = torch.optim.Adam([inputField.data], lr=0.1)
numElems = sensorData[0].data.numel()
smallestLoss = np.Infinity
prevLoss = np.Infinity
numPlateauIterations = 0
fieldsAtSlmPlane_BestFit = []

print('Using optimization to figure out SLM fields...')
print("")
print('    Iteration\t|\tLoss')
print('--------------------------------------')

for t in range(maxIterations):
	curLoss = 0
	# optimizer.zero_grad()
	for idx in range(n_slm_batches):
		slmModel.load_single_slm(batch_idx=idx)
		y = slmInputToSensorOutputModel(inputField)
		# L_fun = torch.sum(((y.data - sensorData[idx].data).abs()) ** 2) / numElems	# Optimization seems slower with this loss function, but this might (?) use less memory.
		L_fun = torch.sum(((y.data.sqrt() - sensorData[idx].data.sqrt()).abs()) ** 2) / numElems

		curLoss = L_fun.item()

		torch.cuda.empty_cache()

		# Old code
			# curLoss = curLoss + L_fun.item()
			# curLoss = curLoss + L_fun

		# # Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		L_fun.backward()
		optimizer.step()

		y.detach()
		del y
		L_fun.detach()
		del L_fun
		torch.cuda.empty_cache()

	# Old code---easy to run out of GPU memory
		# # Averaging to get MSE for entire batch
		# curLoss = curLoss / n_slm_batches

		# # Zero gradients, perform a backward pass, and update the weights.
		# optimizer.zero_grad()
		# curLoss.backward()
		# optimizer.step()

		# curLoss = float(curLoss.cpu())

		# y.detach()
		# del y
		# L_fun.detach()
		# del L_fun
		# torch.cuda.empty_cache()

	print('\t%d\t|\t%.10f' % (t + 1, curLoss))

	if (curLoss < smallestLoss):
		smallestLoss = curLoss
		fieldsAtSlmPlane_BestFit = inputField.data

	# This will terminate the loop if the loss function changes too little for too many iterations
	if (np.abs(curLoss - prevLoss) <= plateauDeltaThreshold):
		numPlateauIterations = numPlateauIterations + 1
	else:
		numPlateauIterations = 0
	if (numPlateauIterations >= plateauIterationsThreshold):
		break
	prevLoss = curLoss

print("")
print('Finished optimization.')
print("")

slmInputPlaneField = ElectricField(
					data = torch.tensor(fieldsAtSlmPlane_BestFit, dtype=torch.cfloat, device=device), # Done this way so that requires_grad=False for this tensor
					wavelengths = sensorData[0].wavelengths.to(device),
					spacing = slmPlanePixelPitch
				)
slmInputPlaneField.spacing = slmInputPlaneField.spacing.to(device)

recoveredSensorPlaneField = slmToSensorPlaneModel(slmInputPlaneField)
	

while True:
	sensorPlaneSavePath = findNoncollidingFilepath(pathlib.Path(), 'SensorPlaneField_' + dataFolderPath.stem, 'pt')
	slmPlaneSavePath = findNoncollidingFilepath(pathlib.Path(), 'SLMPlaneField_' + dataFolderPath.stem, 'pt')
	sensorPlaneSaveFileStr = sensorPlaneSavePath.stem + sensorPlaneSavePath.suffix
	slmPlaneSaveFileStr = slmPlaneSavePath.stem + slmPlaneSavePath.suffix
	resp = input("Save recovered field data as '" + sensorPlaneSaveFileStr + "' and '" + slmPlaneSaveFileStr + "'? (y/n): ")
	if (resp == 'y'):
		torch.save(recoveredSensorPlaneField, sensorPlaneSavePath)
		print("Saved '" + sensorPlaneSaveFileStr + "' to current working directory.")
		torch.save(slmInputPlaneField, slmPlaneSavePath)
		print("Saved '" + slmPlaneSaveFileStr + "' to current working directory.")
		print("Exiting...")
		break
	elif (resp == 'n'):
		print("Exiting...")
		break
	else:
		print("Invalid input.")

pass