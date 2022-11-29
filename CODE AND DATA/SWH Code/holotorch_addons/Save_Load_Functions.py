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
from holotorch.utils.Enumerators import *
from holotorch.CGH_Datatypes.IntensityField import IntensityField

from holotorch_addons.Field_Resampler import Field_Resampler
from holotorch_addons.SimpleDetector import SimpleDetector
from ASM_Prop_Patched import ASM_Prop_Patched

import holotorch_addons.HolotorchPatches as HolotorchPatches

################################################################################################################################




################################################################################################################################

def findNoncollidingFilepath(dir : pathlib.Path, desiredFname : str, fileExtension : str):
	tempName = desiredFname + '.' + fileExtension
	tempPath = dir / tempName
	if not tempPath.exists():
		return tempPath
	for num in range(1, 101):
		tempName = desiredFname + ' (' + str(num) + ').' + fileExtension
		tempPath = dir / tempName
		if not tempPath.exists():
			return tempPath
	raise Exception("Could not find a suitable filename.")


def infer_parameters_from_save_data(dataFolderPath):
	metadataDict = torch.load(dataFolderPath / 'Metadata.pt')
	if (metadataDict['save_version'] == '1.0'):
		return generate_save_data_dict_for_v1_data(dataFolderPath=dataFolderPath)
	else:
		raise Exception("Not implemented.")


def generate_save_data_dict_for_v1_data(dataFolderPath):
	returnDict = {}
	
	metadataDict = torch.load(dataFolderPath / 'Metadata.pt')
	returnDict['metadataFileDict'] = metadataDict

	slmFilenamesTemp = glob.glob(str(dataFolderPath / 'SLM_Data')+"\\*.pt")
	slmFileTemp = pathlib.Path(slmFilenamesTemp[0])
	slmDataTemp = torch.load(slmFileTemp)
	slmFileDataShape = slmDataTemp['_data_tensor'].shape

	validSlmPlanePixelPitchFlag = False
	tempSlmPixelPitch = metadataDict['slmPlanePixelPitch'].data_tensor.squeeze()
	if (len(tempSlmPixelPitch.shape) == 1):
		if (len(tempSlmPixelPitch) == 2):
			if (tempSlmPixelPitch[0] == tempSlmPixelPitch[1]):
				slmPlanePixelPitch = float(tempSlmPixelPitch[0])
				validSlmPlanePixelPitchFlag = True
		elif (len(tempSlmPixelPitch) == 1):
			slmPlanePixelPitch = float(tempSlmPixelPitch[0])
			validSlmPlanePixelPitchFlag = True
	if not (validSlmPlanePixelPitchFlag):
		raise Exception("Invalid format for SLM plane pixel pitch in saved data.  The code is not equipped to handle this case.")
	
	validSensorPlanePixelPitchFlag = False
	tempSensorPixelPitch = metadataDict['sensorPlanePixelPitch'].data_tensor.squeeze()
	if (len(tempSensorPixelPitch.shape) == 1):
		if (len(tempSensorPixelPitch) == 2):
			if (tempSensorPixelPitch[0] == tempSensorPixelPitch[1]):
				sensorPlanePixelPitch = float(tempSensorPixelPitch[0])
				validSensorPlanePixelPitchFlag = True
		elif (len(tempSensorPixelPitch) == 1):
			sensorPlanePixelPitch = float(tempSensorPixelPitch[0])
			validSensorPlanePixelPitchFlag = True
	if not (validSensorPlanePixelPitchFlag):
		raise Exception("Invalid format for sensor plane pixel pitch in saved data.  The code is not equipped to handle this case.")

	returnDict['slmPlaneResolution'] = tuple(slmFileDataShape[-2:])
	returnDict['slmPlanePixelPitch'] = slmPlanePixelPitch
	returnDict['numChannels'] = slmFileDataShape[2]		# Since SLMs use BTCHW dimensions, the channel dimension is the 3rd dimension (index #2)
	returnDict['n_batch'] = slmFileDataShape[0] * len(slmFilenamesTemp)
	returnDict['n_slm_batches'] = len(slmFilenamesTemp)

	sensorFilenamesTemp = glob.glob(str(dataFolderPath / 'Sensor_Data')+"\\*.pt")
	sensorFileTemp = pathlib.Path(sensorFilenamesTemp[0])
	sensorDataTemp = torch.load(sensorFileTemp)
	sensorFileDataShape = sensorDataTemp['data'].shape

	slmFileDataShape6D = torch.zeros(slmFileDataShape)[:,:,None,:,:,:].shape
	if (sensorFileDataShape[0:4] != slmFileDataShape6D[0:4]):
		raise Exception("Differing numbers of batches and/or channels for SLM and sensor data mini-batches.")
	if ((sensorFileDataShape[0] * len(sensorFilenamesTemp)) != (slmFileDataShape[0] * len(slmFilenamesTemp))):
		raise Exception("The total sizes of the batch dimension for the SLMs and sensor data are different.")

	returnDict['sensorPlaneResolution'] = tuple(sensorFileDataShape[-2:])
	returnDict['sensorPlanePixelPitch'] = sensorPlanePixelPitch
	returnDict['wavelengths'] = sensorDataTemp['wavelengths']
	returnDict['sensorPlaneSpacing'] = sensorDataTemp['spacing']

	return returnDict


# Assumes the model: SLM --> ASM_Prop --> (Field_Resampler) --> Sensor (Detector)
def recreate_slm_to_sensor_plane_model_for_v1_data(loadedInfo : dict, saveDataFolderPath : pathlib.Path, tempSlmDataFolderPath : pathlib.Path, storeSlmTempDataOnGpu : bool = False, slmToSensorPlaneDistance : float = None, device : torch.device = None):
	sensorPlaneResolution = loadedInfo['sensorPlaneResolution']
	slmPlaneResolution = loadedInfo['slmPlaneResolution']
	slmPlanePixelPitch = loadedInfo['slmPlanePixelPitch']
	sensorPlanePixelPitch = loadedInfo['sensorPlanePixelPitch']
	numChannels = loadedInfo['numChannels']
	numSpecklePatternsPerWavelength = loadedInfo['n_batch']
	numSlmMiniBatches = loadedInfo['n_slm_batches']

	if ('additional_data' in loadedInfo['metadataFileDict']):
		if ('slmToSensorPlaneDistance' in loadedInfo['metadataFileDict']['additional_data']):
			if (slmToSensorPlaneDistance is None):
				slmToSensorPlaneDistance = loadedInfo['metadataFileDict']['additional_data']['slmToSensorPlaneDistance']
			else:
				# Do nothing as the value of 'slmToSensorPlaneDistance' should have already been set
				warnings.warn("Data has specified 'slmToSensorPlaneDistance' but a value for that parameter was provided as an argument.  Overriding the value from the data.")
	if (slmToSensorPlaneDistance is None):
		raise Exception("Could not determine what the SLM-to-sensor-plane distance was---this means that it was not given in the saved data or specified as an argument to the function that this error is raised in.")

	slmModel =	HolotorchPatches.SLM_PhaseOnly_Patched.create_slm(	height          		= slmPlaneResolution[0],
																	width           		= slmPlaneResolution[1],
																	n_channel       		= numChannels,
																	n_batch					= numSpecklePatternsPerWavelength,
																	n_slm_batches			= numSlmMiniBatches,
																	feature_size    		= slmPlanePixelPitch,
																	init_type       		= ENUM_SLM_INIT.RANDOM, # This should not matter as we are loading SLM data
																	slm_directory			= tempSlmDataFolderPath.resolve(),
																	static_slm				= True,
																	static_slm_data_path	= saveDataFolderPath / 'SLM_Data',
																	store_on_gpu			= storeSlmTempDataOnGpu,
																	device					= device
																)
	slmToSensorPlanePropASM = ASM_Prop_Patched(init_distance = slmToSensorPlaneDistance)
	slmResToSensorResResampler = 	Field_Resampler(
										outputHeight = sensorPlaneResolution[0],
										outputWidth = sensorPlaneResolution[1],
										outputPixel_dx = sensorPlanePixelPitch,
										outputPixel_dy = sensorPlanePixelPitch,
										device = device
									)
	slmToSensorPlaneModel = torch.nn.Sequential(slmToSensorPlanePropASM, slmResToSensorResResampler)
	detectorModel = SimpleDetector()
	slmInputToSensorOutputModel = torch.nn.Sequential(slmModel, slmToSensorPlaneModel, detectorModel)

	return slmInputToSensorOutputModel, slmModel, slmToSensorPlaneModel, detectorModel


def loadSensorData(dataFolderPath : pathlib.Path, device : torch.device = None):
	sensorFilePathsTemp = glob.glob(str(dataFolderPath / 'Sensor_Data')+"\\*.pt")
	numFiles = len(sensorFilePathsTemp)
	sensorData = [None] * numFiles
	
	for k in range(numFiles):
		filepathTemp = pathlib.Path(sensorFilePathsTemp[k])
		dataTemp = torch.load(filepathTemp, map_location=device)
		
		tempData = dataTemp['data']
		tempWavelengths = dataTemp['wavelengths']
		tempSpacing = dataTemp['spacing']

		tempField =	IntensityField(
						data = tempData,
						wavelengths = tempWavelengths,
						spacing = tempSpacing
					)

		sensorData[k] = tempField
	
	return sensorData