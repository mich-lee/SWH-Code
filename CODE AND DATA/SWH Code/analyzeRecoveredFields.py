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

from holotorch_addons.HelperFunctions import computeSpatialFrequencyGrids, computeBandlimitASM, computeBandlimitingFilterSpaceDomain, computeBandlimitingFilterASM, get_field_slice, print_cuda_memory_usage, generateGrid
from holotorch_addons.Field_Resampler import Field_Resampler
from holotorch_addons.SimpleDetector import SimpleDetector
# from holotorch_addons.Thin_Lens import Thin_Lens

import holotorch_addons.HolotorchPatches as HolotorchPatches
from Synthetic_Data_Generator import Synthetic_Data_Generator

################################################################################################################################




################################################################################################################################
def recoverAnglesAndDepthMap(	field : ElectricField or torch.tensor,
								wavelengthInput : float = None,
								noiseAmplitudeCutoff : float = -np.Infinity,	# Any point in the field that has an absolute value of <= noiseAmplitudeCutoff will be assigned a depth of noiseDepthValue
								# noiseDepthValue : float = 0,
								attemptToRecenterPhases : bool = False,
								verbose : bool = False
							):

	if (type(field) is ElectricField):
		fieldData = field.data
	elif (torch.is_tensor(field)):
		fieldData = field
	else:
		raise Exception("'field' argument must be a PyTorch tensor or an ElectricField object.")

	if (wavelengthInput is None):
		if (type(field) is ElectricField):
			if (field.wavelengths.data_tensor.squeeze().ndim != 0):
				raise Exception("Could not determine wavelength.  The 'field' argument is an ElectricField object that has multiple wavelengths associated with it.  The 'wavelengthInput' argument has a value of None so cannot default to that.")
			else:
				wavelength = float(field.wavelengths.data_tensor.squeeze())
		else:
			raise Exception("Could not determine wavelength.  The 'field' argument is a PyTorch tensor.  The 'wavelengthInput' argument has a value of None so cannot default to that.")
	else:
		wavelength = float(wavelengthInput)

	if not attemptToRecenterPhases:
		fieldDataAngles = fieldData.angle()
	else:
		# phaseShift = torch.tensor(0, dtype=torch.float, device=fieldData.device, requires_grad=True)
		phaseShift = torch.tensor(fieldData[fieldData.abs() >= noiseAmplitudeCutoff].angle().mean(), dtype=torch.float, device=fieldData.device, requires_grad=True)
		optimizer = torch.optim.Adam([phaseShift], lr=0.1)
		maxIterations = 500
		plateauDeltaThreshold = 1e-7
		plateauIterationsThreshold = 20
		numElems = fieldData.numel()
		smallestLoss = np.Infinity
		prevLoss = np.Infinity
		numPlateauIterations = 0
		bestPhaseShift = 0

		angleTemp = fieldData[fieldData.abs() >= noiseAmplitudeCutoff].angle()	# Ignoring phases of points with small amplitudes

		if verbose:
			print('Using optimization to recenter phases...')
			print("")
			print('    Iteration\t|\tLoss (MSE)')
			print('--------------------------------------')

		for t in range(maxIterations):
			L_fun = (torch.exp(1j * (angleTemp + phaseShift)).angle() ** 2).sum() / numElems		# Loss function measures variance about zero
			curLoss = L_fun.item()

			optimizer.zero_grad()
			L_fun.backward()
			optimizer.step()

			if (curLoss < smallestLoss):
				smallestLoss = curLoss
				bestPhaseShift = phaseShift

			if verbose:
				print('\t%d\t|\t%.10f' % (t + 1, curLoss))

			# This will terminate the loop if the loss function changes too little for too many iterations
			if (np.abs(curLoss - prevLoss) <= plateauDeltaThreshold):
				numPlateauIterations = numPlateauIterations + 1
			else:
				numPlateauIterations = 0
			if (numPlateauIterations >= plateauIterationsThreshold):
				break
			prevLoss = curLoss

		fieldDataAngles = (fieldData * torch.exp(1j*bestPhaseShift)).angle()

	depthMapTemp = fieldDataAngles * (1/(2*np.pi)) * wavelength
	# depthMapTemp[fieldData.abs() < noiseAmplitudeCutoff] = noiseDepthValue
	depthMapTemp[fieldData.abs() < noiseAmplitudeCutoff] = np.NaN
	fieldDataAngles[fieldData.abs() < noiseAmplitudeCutoff] = np.NaN

	return fieldDataAngles.detach(), depthMapTemp.detach()


def recoverDepthMap(	field : ElectricField or torch.tensor,
						wavelengthInput : float = None,
						noiseAmplitudeCutoff : float = -np.Infinity,	# Any point in the field that has an absolute value of <= noiseAmplitudeCutoff will be assigned a depth of noiseDepthValue
						attemptToRecenterPhases : bool = False,
						verbose : bool = False
					):
	_, depthMap = recoverAnglesAndDepthMap(field=field, wavelengthInput=wavelengthInput, noiseAmplitudeCutoff=noiseAmplitudeCutoff, attemptToRecenterPhases=attemptToRecenterPhases, verbose=verbose)
	return depthMap

################################################################################################################################




################################################################################################################################
use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
################################################################################################################################




################################################################################################################################
# fields = torch.load("../DATA/Recovered Fields/SensorPlaneField_DATA_2022-9-11_16h18m11s.pt")
# fields = torch.load("../DATA/Recovered Fields/SensorPlaneField_DATA_2022-9-13_02h43m19s.pt")
# fields = torch.load("../DATA/Recovered Fields/SensorPlaneField_DATA_2022-9-18_07h06m11s.pt")
fields = torch.load("../DATA/Recovered Fields/SensorPlaneField_DATA_2022-10-2_20h22m28s.pt")

fields.data = fields.data.to(device)
fields.wavelengths = fields.wavelengths.to(device)
fields.spacing = fields.spacing.to(device)


swhFields = [[None for j in range(fields.num_channels)] for i in range(fields.num_channels)]
swhWavelengths = [[None for j in range(fields.num_channels)] for i in range(fields.num_channels)]

for i in range(fields.num_channels):
	for j in range(fields.num_channels):
		if (i != j):
			l1 = float(fields.wavelengths.data_tensor[i])
			l2 = float(fields.wavelengths.data_tensor[j])
			wavelengthTemp = l1 * l2 / abs(l1 - l2)		# Make wavelength object with wavelength being a synthetic wavelength
														# The recoverAnglesAndDepthMap(...) and recoverDepthMap(...) functions will make use of this synthetic wavelength in their calculations.
			dataTemp = fields.data[0,0,0,i,:,:] * fields.data[0,0,0,j,:,:].conj()
			dataTemp = dataTemp[None,None,None,None,:,:]
		else:
			wavelengthTemp = float(fields.wavelengths.data_tensor[i])
			dataTemp = fields.data[0,0,0,i,:,:]
		spacingTemp = float(fields.spacing.data_tensor[0,0,0])

		E_temp = ElectricField(data=dataTemp, wavelengths=wavelengthTemp, spacing=spacingTemp)
		# E_temp.data = E_temp.data.to(device)	# Not needed as the data in 'fields' should already be on the correct device
		E_temp.wavelengths = E_temp.wavelengths.to(device)
		E_temp.spacing = E_temp.spacing.to(device)

		swhFields[i][j] = E_temp
		swhWavelengths[i][j] = wavelengthTemp


selectedInd_i = 1
selectedInd_j = 0
noiseAmplitudeCutoff = 0.5
attemptToRecenterPhases = True

swhFieldTemp = swhFields[selectedInd_i][selectedInd_j]
# depthMapTemp = recoverDepthMap(swhFieldTemp, noiseAmplitudeCutoff=noiseAmplitudeCutoff, attemptToRecenterPhases=attemptToRecenterPhases)
swhAnglesTemp, depthMapTemp = recoverAnglesAndDepthMap(swhFieldTemp, noiseAmplitudeCutoff=noiseAmplitudeCutoff, attemptToRecenterPhases=attemptToRecenterPhases)#, verbose=True)
xGrid, yGrid = generateGrid(tuple(depthMapTemp.shape[-2:]), swhFieldTemp.spacing.data_tensor[0,0,0].to('cpu'), swhFieldTemp.spacing.data_tensor[0,0,1].to('cpu'))
depthMapSaveDict = 	{
						'depth' : depthMapTemp,
						'xGrid' : xGrid,
						'yGrid' : yGrid,
						'lambda_1' : fields.wavelengths.data_tensor[selectedInd_i],
						'lambda_2' : fields.wavelengths.data_tensor[selectedInd_j],
						'lambda_synth' : swhWavelengths[selectedInd_i][selectedInd_j],
						'noiseAmplitudeCutoff' : noiseAmplitudeCutoff,
						'phasesRecentered' : attemptToRecenterPhases
					}

dx = xGrid[1,0] - xGrid[1,0]
dy = yGrid[1,0] - yGrid[1,0]
imshowPlotExtent = (yGrid.min() - dy/2, yGrid.max() + dy/2, xGrid.min() - dx/2, xGrid.max() + dx/2)

plt.clf()
plt.subplot(1,2,1)
tempPlotData1 = swhAnglesTemp.clone()
tempPlotData1[swhAnglesTemp.isnan()] = 0
plt.imshow(tempPlotData1.squeeze().cpu(), extent=imshowPlotExtent)
plt.xlim(-0.0008,0.0008)
plt.ylim(-0.0008,0.0008)
plt.colorbar()
plt.subplot(1,2,2)
tempPlotData2 = depthMapTemp.clone()
tempPlotData2[depthMapTemp.isnan()] = 0
plt.imshow(tempPlotData2.squeeze().cpu(), extent=imshowPlotExtent)
plt.xlim(-0.0008,0.0008)
plt.ylim(-0.0008,0.0008)
plt.colorbar()

# x0_ind = 1000
# x1_ind = 2000
# y0_ind = 1500
# y1_ind = 2500
# downsampleJump = 10
# depthMapDownsampled = depthMapTemp[... , x0_ind:x1_ind:downsampleJump, y0_ind:y1_ind:downsampleJump] # depthMapTemp[... , 0::16, 0::16]
# xGridDownsampled = xGrid[x0_ind:x1_ind:downsampleJump, y0_ind:y1_ind:downsampleJump] # xGrid[0::16, 0::16]
# yGridDownsampled = yGrid[x0_ind:x1_ind:downsampleJump, y0_ind:y1_ind:downsampleJump] # yGrid[0::16, 0::16]
# plt.clf()
# ax = plt.axes(projection='3d')
# ax.plot_surface(xGridDownsampled, yGridDownsampled, depthMapDownsampled.squeeze().cpu(), rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=1)

1 == 1

while True:
	curDateTime = datetime.datetime.today()
	saveFileStr = 'DepthMap_' + str(selectedInd_i) + '_' + str(selectedInd_j) + '_' + str(curDateTime.year) + '-' + str(curDateTime.month) + '-' + str(curDateTime.day) + '_' + \
						str(curDateTime.hour).zfill(2) + 'h' + str(curDateTime.minute).zfill(2) + 'm' + str(curDateTime.second).zfill(2) + 's' + '.pt'
	resp = input("Save depth map data as '" + saveFileStr + "'? (y/n): ")
	if (resp == 'y'):
		torch.save(depthMapSaveDict, saveFileStr)
		print("Saved file.")
		print("Exiting...")
		break
	elif (resp == 'n'):
		print("Exiting...")
		break
	else:
		print("Invalid input.")

pass