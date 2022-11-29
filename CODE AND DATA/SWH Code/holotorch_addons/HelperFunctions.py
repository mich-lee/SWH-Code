# from logging import exception
import string
import wave
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

import copy
import re
import pathlib

# Image wranglers
import imageio
from PIL import Image

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.Dimensions import *
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.CGH_Datatypes.Light import Light
from holotorch.CGH_Datatypes.IntensityField import IntensityField

from holotorch.utils.Helper_Functions import ft2, ift2


def print_cuda_memory_usage(device : torch.device, printShort = False):
	gpu_mem_allocated = torch.cuda.memory_allocated(device)
	gpu_mem_reserved = torch.cuda.memory_reserved(device)
	gpu_mem_total = torch.cuda.get_device_properties(device).total_memory
	if not printShort:
		gpu_info_printout_lines = torch.cuda.memory_summary(device=device,abbreviated=True).split('\n')
		gpu_info_printout_str = '\n'.join([gpu_info_printout_lines[i] for i in [0,1,4,5,6,7,11,17,26]])
		print(gpu_info_printout_str)
		print('  Memory Usage (Reserved): %.2f GB / %.2f GB  -  %.2f%%' % (gpu_mem_reserved/(1024**3), gpu_mem_total/(1024**3), (gpu_mem_reserved/gpu_mem_total)*100))
	else:
		print('  Allocated: %.2f GB\tReserved: %.2f GB\tTotal: %.2f GB' % (gpu_mem_allocated/(1024**3), gpu_mem_reserved/(1024**3), gpu_mem_total/(1024**3)))


def get_tensor_size_bytes(tensorInput : torch.Tensor):
	return tensorInput.nelement() * tensorInput.element_size()


def generateGrid(res, deltaX, deltaY, centerGrids = True, centerAroundZero = True, device=None):
	if (torch.is_tensor(deltaX)):
		deltaX = copy.deepcopy(deltaX).squeeze().to(device=device)
	if (torch.is_tensor(deltaY)):
		deltaY = copy.deepcopy(deltaY).squeeze().to(device=device)

	if (centerGrids):
		if (centerAroundZero):
			xCoords = torch.linspace(-((res[0] - 1) // 2), (res[0] - 1) // 2, res[0]).to(device=device) * deltaX
			yCoords = torch.linspace(-((res[1] - 1) // 2), (res[1] - 1) // 2, res[1]).to(device=device) * deltaY
		else:
			xCoords = (torch.linspace(0, res[0] - 1, res[0]) - (res[0] // 2)).to(device=device) * deltaX
			yCoords = (torch.linspace(0, res[1] - 1, res[1]) - (res[1] // 2)).to(device=device) * deltaY
	else:
		xCoords = torch.linspace(0, res[0] - 1, res[0]).to(device=device) * deltaX
		yCoords = torch.linspace(0, res[1] - 1, res[1]).to(device=device) * deltaY

	xGrid, yGrid = torch.meshgrid(xCoords, yCoords)

	return xGrid, yGrid


# Resizes image while keeping its aspect ratio.  Will make the resized image as big as possible without
# exceeding the resolution set by 'targetResolution'.
#	- For example, if the target resolution is 600x400 pixels, a 200x200 pixel input image will be resized to 400x400 pixels.
def fit_image_to_resolution(inputImage, targetResolution):
	def getNumChannels(shape):
		if (len(shape) == 2):
			return 1
		elif (len(shape) == 3):
			return shape[2]
		else:
			raise Exception("Unrecognized image data shape.")

	inputImageAspectRatio = inputImage.size[0] / inputImage.size[1]
	targetAspectRatio = targetResolution[1] / targetResolution[0]

	if (targetAspectRatio == inputImageAspectRatio):
		outputImage = inputImage.resize((targetResolution[1], targetResolution[0]))
	elif (inputImageAspectRatio < targetAspectRatio):
		# Width relatively undersized so should resize to match height
		imageMag = targetResolution[0] / inputImage.size[1]																		# = (Target resolution height) / (Input image height)
		imageMagWidth = np.int(np.floor(inputImage.size[0] * imageMag))															# = floor((Input image width) * imageMag)
		resizedImageData = np.asarray(inputImage.resize((imageMagWidth, targetResolution[0])))									# Resize input image to match target resolution's height, then convert image to array
		paddingOffset = (targetResolution[1] - imageMagWidth) // 2																# Calculate how much more width the target resolution has relative to the input image, divide that number by 2, and round down
		numChannels = getNumChannels(resizedImageData.shape)
		paddedImageData = np.zeros([targetResolution[0], targetResolution[1], numChannels])										# Initialize new array for image data (array indices represent height, width, and color/alpha channels respectively)
		if (numChannels == 1):
			resizedImageData = resizedImageData[:,:,None]
		paddedImageData[:,paddingOffset:(paddingOffset+imageMagWidth),:] = resizedImageData										# Put resizedImageData array into paddedImageData, with the resizedImageData array being centered in the width dimension
		if (numChannels == 1):
			paddedImageData = np.squeeze(paddedImageData)
			outputImage = Image.fromarray(paddedImageData.astype(np.uint8), mode='L')
		else:
			outputImage = Image.fromarray(paddedImageData.astype(np.uint8))														# Convert paddedImageData to an image object
	else:
		# Height relatively undersized so should resize to match width
		imageMag = targetResolution[1] / inputImage.size[0]																		# = (Target resolution width) / (Input image width)
		imageMagHeight = np.int(np.floor(inputImage.size[1] * imageMag))														# = floor((Input image height) * imageMag)
		resizedImageData = np.asarray(inputImage.resize((targetResolution[1], imageMagHeight)))									# Resize input image to match target resolution's width, then convert image to array
		paddingOffset = (targetResolution[0] - imageMagHeight) // 2																# Calculate how much more height the target resolution has relative to the input image, divide that number by 2, and round down
		numChannels = getNumChannels(resizedImageData.shape)
		paddedImageData = np.zeros([targetResolution[0], targetResolution[1], numChannels])										# Initialize new array for image data (array indices represent height, width, and color/alpha channels respectively)
		if (numChannels == 1):
			resizedImageData = resizedImageData[:,:,None]
		paddedImageData[paddingOffset:(paddingOffset+imageMagHeight),:,:] = resizedImageData									# Put resizedImageData array into paddedImageData, with the resizedImageData array being centered in the height dimension
		if (numChannels == 1):
			paddedImageData = np.squeeze(paddedImageData)
			outputImage = Image.fromarray(paddedImageData.astype(np.uint8), mode='L')
		else:
			outputImage = Image.fromarray(paddedImageData.astype(np.uint8))														# Convert paddedImageData to an image object

	return outputImage


def parseNumberAndUnitsString(str):
	unitsMatches = re.findall('(nm)|(um)|(mm)|(cm)|(ms)|(us)|(ns)|(m)|(s)', str)
	unitStrings = ['nm', 'um', 'mm', 'cm', 'ms', 'us', 'ns', 'm', 's']
	unitTypes = ['spatial', 'spatial', 'spatial', 'spatial', 'time', 'time', 'time', 'spatial', 'time']
	unitsMultipliers = [1e-9, 1e-6, 1e-3, 1e-2, 1e-3, 1e-6, 1e-9, 1, 1]
	if (len(unitsMatches) > 1):
		raise Exception("Invalid number string.")
	elif (len(unitsMatches) == 0):
		multiplier = 1
		numStr = str
		unitStr = ''
		unitTypeStr = ''
	else: # len(unitsMatches) == 1
		unitStr = ''.join(list(unitsMatches[-1]))
		unitIndex = unitStrings.index(unitStr)
		multiplier = unitsMultipliers[unitIndex]
		unitTypeStr = unitTypes[unitIndex]
		numStr = str[0:-len(unitStr)]
	try:
		return float(numStr) * multiplier, unitStr, unitTypeStr
	except:
		raise Exception("Invalid number string.")


# Not bothering to check for matching dimensions here
def applyFilterSpaceDomain(h : torch.tensor, x : torch.tensor):
	# Assumes h and x have the same size
	Nx_old = int(x.shape[-2])
	Ny_old = int(x.shape[-1])

	pad_nx = int(Nx_old / 2)
	pad_ny = int(Ny_old / 2)

	# As of 9/2/2022, cannot rely on padding in ft2(...) function
	# That function has the pad_nx and pad_ny arguments reversed in its call to torch.nn.functional.pad(...)
	# Because of that, the x dimension (height) gets y's padding amount and vice-versa.
	# Therefore, doing the padding here.
	hPadded = torch.nn.functional.pad(h, (pad_ny,pad_ny,pad_nx,pad_nx), mode='constant', value=0)
	xPadded = torch.nn.functional.pad(x, (pad_ny,pad_ny,pad_nx,pad_nx), mode='constant', value=0)
	H = ft2(hPadded, pad=False)
	X = ft2(xPadded, pad=False)

	Y = H * X
	y = ift2(Y)
	y = y[..., pad_nx:(pad_nx+Nx_old), pad_ny:(pad_ny+Ny_old)]

	return y


# Not bothering to check for matching dimensions here
def applyFilterToElectricField(h : torch.tensor, field : ElectricField):
	filteredField = ElectricField(
						data = applyFilterSpaceDomain(h, field.data),
						wavelengths = field.wavelengths,
						spacing = field.spacing
					)
	return filteredField


# This function uses some code from ASM_Prop in the HoloTorch library
def getWavelengthAndSpacingDataAsTCHW(field : ElectricField):
	# extract dx, dy spacing into TxC tensors
	spacing = field.spacing.data_tensor
	dx      = spacing[:,:,0]
	if spacing.shape[2] > 1:
		dy = spacing[:,:,1]
	else:
		dy = dx

	# Get wavelengths as TxCxHxW tensors
	wavelengths = field.wavelengths
	new_shape       = wavelengths.tensor_dimension.get_new_shape(new_dim=Dimensions.TC) # Get TxC shape
	wavelengths_TC  = wavelengths.data_tensor.view(new_shape) # Reshape to TxC tensor
	wavelengths_TC  = wavelengths_TC[:,:,None,None] # Expand wavelengths for H and W dimension
	
	# Get dx and dy as TxCxHxW tensors
	dx_TC   = dx.expand(new_shape) # Reshape to TxC tensor
	dx_TC   = dx_TC[:,:,None,None] # Expand to H and W
	dy_TC   = dy.expand(new_shape) # Reshape to TxC tensor
	dy_TC   = dy_TC[:,:,None,None] # Expand to H and W
	
	return wavelengths_TC, dx_TC, dy_TC


# This function uses some code from ASM_Prop in the HoloTorch library
def computeSpatialFrequencyGrids(field : ElectricField):
	_, dx_TC, dy_TC = getWavelengthAndSpacingDataAsTCHW(field)

	# Initializing return values
	Kx = None
	Ky = None

	# Want values on the half-open interval [-1/2, 1/2).  Want to exclude +1/2 as it is redundant with -1/2
	Kx_vals_normed = torch.linspace(-np.floor(field.height / 2), np.floor((field.height - 1) / 2), field.height) / field.height
	Ky_vals_normed = torch.linspace(-np.floor(field.width / 2), np.floor((field.width - 1) / 2), field.width) / field.width
	
	# Making grid and ensuring that it is on the correct device
	Kx_Grid, Ky_Grid = torch.meshgrid(Kx_vals_normed, Ky_vals_normed)
	Kx_Grid = Kx_Grid.to(dx_TC.device)
	Ky_Grid = Ky_Grid.to(dy_TC.device)

	# Expand the frequency grid for T and C dimension
	Kx = 2*np.pi * Kx_Grid[None,None,:,:] / dx_TC
	Ky = 2*np.pi * Ky_Grid[None,None,:,:] / dy_TC

	return Kx, Ky


# This function uses some code from ASM_Prop in the HoloTorch library
def computeBandlimitASM(field : ElectricField, zPropDistance : float):
	wavelengths_TC, dx_TC, dy_TC = getWavelengthAndSpacingDataAsTCHW(field)

	# size of the field
	# # Total field size on the hologram plane
	length_x = field.height * dx_TC 
	length_y = field.width  * dy_TC

	# band-limited ASM - Matsushima et al. (2009)	|	Paper: "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields"
	f_y_max = 2*np.pi / torch.sqrt((2 * zPropDistance * (1 / length_x) ) **2 + 1) / wavelengths_TC
	f_x_max = 2*np.pi / torch.sqrt((2 * zPropDistance * (1 / length_y) ) **2 + 1) / wavelengths_TC

	return f_x_max, f_y_max


# This function uses some code from ASM_Prop in the HoloTorch library
def computeBandlimitingFilterSpaceDomain(f_x_max, f_y_max, Kx, Ky):
	# Should be 4D T x C x H x W, thus ift2(...) can be used.
	bandlimiting_Filter_Freq = torch.zeros_like(Kx)
	bandlimiting_Filter_Freq[ ( torch.abs(Kx) < f_x_max) & (torch.abs(Ky) < f_y_max) ] = 1
	bandlimiting_Filter_Space = ift2(bandlimiting_Filter_Freq)
	bandlimiting_Filter_Space = bandlimiting_Filter_Space[None,:,None,:,:,:] # Expand from 4D to 6D (TCHW dimensions --> BTPCHW dimensions)
	return bandlimiting_Filter_Space


def computeBandlimitingFilterASM(wavelengthsArray, slmPlaneResolution, slmPlanePixelPitch, propagationDistance, device = None):
	slmInputFieldSize = torch.Size([1, 1, 1, np.size(wavelengthsArray), slmPlaneResolution[0], slmPlaneResolution[1]])
	
	if (device is None):
		dummyData = torch.ones(slmInputFieldSize, dtype=torch.cfloat)
	else:
		dummyData = torch.ones(slmInputFieldSize, dtype=torch.cfloat, device=device)
	
	slmInputFieldPrototype = ElectricField(
								data = dummyData,
								wavelengths = WavelengthContainer(wavelengths=wavelengthsArray, tensor_dimension=Dimensions.C(n_channel=np.size(wavelengthsArray))),
								spacing = slmPlanePixelPitch
							)
	
	Kx, Ky = computeSpatialFrequencyGrids(slmInputFieldPrototype)
	f_x_max, f_y_max = computeBandlimitASM(slmInputFieldPrototype, propagationDistance)
	bandlimitingFilterASM = computeBandlimitingFilterSpaceDomain(f_x_max, f_y_max, Kx, Ky)

	return bandlimitingFilterASM


# Explanation:
#	Given TensorDimensions old_dim and new_dim, this function returns a list of dimension indices from old_dim and a
# list of their corresponding dimension indices in new_dim.
#
# Example:
#	- Suppose old_dim had dimensions ['T','P','C'] and new_dim had dimensions ['B','T','C','H','W'].  Then, this function will return:
#		- old_indices = tensor([0, 2])
#		- new_indices = tensor([1, 2])
def get_dimension_inds_remapping(old_dim : TensorDimension, new_dim : TensorDimension):
	"""[summary]

	Args:
		dim (TensorDimension): [description]
		new_dim (TensorDimension): [description]
	"""
	new_indices_bool = np.isin(new_dim.id, old_dim.id)
	old_indices_bool = np.isin(old_dim.id, new_dim.id)

	new_indices = torch.tensor(range(len(new_dim.id)))[new_indices_bool]
	old_indices = torch.tensor(range(len(old_dim.id)))[old_indices_bool]
	
	return old_indices, new_indices


def check_tensors_broadcastable(a : torch.Tensor, b : torch.Tensor):
	if not (isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor)):
		raise Exception("Error: Need tensor inputs for check_tensors_broadcastable(...).")
	lenDiff = abs(len(a.shape) - len(b.shape))
	for i in range(min(len(a.shape), len(b.shape)) - 1, -1, -1):
		aInd = i + lenDiff*(len(a.shape) > len(b.shape))
		bInd = i + lenDiff*(len(a.shape) < len(b.shape))
		if not ((a.shape[aInd] == 1) or (b.shape[bInd] == 1) or (a.shape[aInd] == b.shape[bInd])):
			return False
	return True


# Returns a field object that is a subset of the input field.
# Can use this to pick out certain batches, channels, regions of 2D space, etc
def get_field_slice(field : Light,
					batch_inds_range : int = None,
					time_inds_range : int = None,
					pupil_inds_range : int = None,
					channel_inds_range : int = None,
					height_inds_range : int = None,
					width_inds_range : int = None,
					field_data_tensor_dimension : TensorDimension = None, 	# For specifying the field's data tensor dimension in case it cannot
																			# be inferred from field.wavelengths.
					cloneTensors : bool = True,		# Could probably get away with this being false for many cases.
													# However, setting this to true will help assure one that data in the input argument 'field' will not get modified by this method.
													# I am not 100% sure though whether such is possible in this method.
					device : torch.device = None
				):

	def checkIfIndicesValid(inds, dimSize):
		if (dimSize == 0):
			return False
		if (isinstance(inds,tuple)):
			if (len(inds) != 2):
				if not (isinstance(inds[0], int) and isinstance(inds[1], int)):
					return False
				if (inds[0] >= dimSize) or (inds[0] < 0):
					return False
				if (inds[1] > dimSize) or (inds[1] <= 0):
					return False
				if (inds[0] >= inds[1]):
					return False
		elif (isinstance(inds, int)):
			ind = inds
			if (ind >= dimSize) or (ind < 0):
				return False
		elif (inds is None):
			return True
		else:
			return False
		return True

	def getDimIndicesList(inds, curDimSize, maxDimSize):
		if not (checkIfIndicesValid(inds, maxDimSize)):
			raise Exception("Invalid indices given when trying to slice a field object!  Indices should be given in the form of an integer or a 2-tuple.")
		if (inds is None):
			return torch.tensor(range(0, min(curDimSize, maxDimSize)))
		if (curDimSize == 1):
			return torch.tensor([0])
		if (isinstance(inds,int)):
			return torch.tensor([inds])
		return torch.tensor(range(inds[0], inds[1]))
			

	if (cloneTensors):
		field_data = torch.clone(field.data)
	else:
		field_data = field.data
	wavelengths = field.wavelengths
	spacing = field.spacing

	if (device is None):
		device = field.data.device


	fieldBTPCHW_shape = torch.ones(6,dtype=int)
	if (field_data_tensor_dimension is None):
		# Trying to infer dimension labels on field_data using the wavelength container
		old_indices, new_indices = get_dimension_inds_remapping(old_dim=wavelengths.tensor_dimension, new_dim=Dimensions.BTPCHW)
		if ((len(field_data.shape) - 2) != len(wavelengths.tensor_dimension.id)):
			raise Exception("ERROR: Could not infer field data dimension labels from wavelength container.  Please manually specify with the 'field_data_tensor_dimension' argument.")
		fieldBTPCHW_shape[new_indices.tolist()] = torch.tensor(field_data.shape)[old_indices.tolist()]
		# Assuming that the wavelengths container does not have height and width dimensions...
		fieldBTPCHW_shape[4:6] = torch.tensor([field_data.shape[-2], field_data.shape[-1]])
		fieldBTPCHW_shape = torch.Size(fieldBTPCHW_shape)
	else:
		old_indices, new_indices = get_dimension_inds_remapping(old_dim=field_data_tensor_dimension, new_dim=Dimensions.BTPCHW)
		fieldBTPCHW_shape[new_indices.tolist()] = torch.tensor(field_data.shape)[old_indices.tolist()]
		fieldBTPCHW_shape = torch.Size(fieldBTPCHW_shape)

	wavelengthsBTPCHW_shape = wavelengths.tensor_dimension.get_new_shape(new_dim=Dimensions.BTPCHW)

	# The last dimension for spacing containers should be height, which is used to hold x and y spacing.
	# Want to split up the last dimension into two so it works with the other tensors more nicely
	if (cloneTensors):
		spacingDataX = torch.clone(spacing.data_tensor)[... , 0].unsqueeze(-1)
		spacingDataY = torch.clone(spacing.data_tensor)[... , 1].unsqueeze(-1)
	else:
		spacingDataX = spacing.data_tensor[... , 0].unsqueeze(-1)
		spacingDataY = spacing.data_tensor[... , 1].unsqueeze(-1)
	
	# Halving the height component because we want to work with one spacing coordinate at a time for now
	spacingBTPCHW_shape = spacing.tensor_dimension.get_new_shape(new_dim=Dimensions.BTPCHW)	# As of 9/7/2022, get_new_shape(...) should not modify any of its arguments.
	spacingBTPCHW_shape = torch.tensor(spacingBTPCHW_shape)	# torch.tensor(...) copies data
	spacingBTPCHW_shape[-2] = 1
	spacingBTPCHW_shape = torch.Size(spacingBTPCHW_shape)

	
	maxDim_shape = torch.Size(torch.maximum(
												torch.maximum(
																torch.tensor(fieldBTPCHW_shape),
																torch.tensor(wavelengthsBTPCHW_shape)	# This adds dimensions for height and width at the end
																														# It's being assumed that the wavelengths dimensions do not include height and width
															),
												torch.tensor(spacingBTPCHW_shape)
											)
							)

	# NOTE: Have not tested this for fields with BTCHW_E dimensions
	if (cloneTensors):
		fieldBTPCHW = torch.clone(field_data).view(fieldBTPCHW_shape).to(device)
		wavelengthsBTPCHW = torch.clone(wavelengths.data_tensor).view(wavelengthsBTPCHW_shape).to(device)
	else:
		fieldBTPCHW = field_data.view(fieldBTPCHW_shape).to(device)
		wavelengthsBTPCHW = wavelengths.data_tensor.view(wavelengthsBTPCHW_shape).to(device)
	spacingX_BTPCHW = spacingDataX.view(spacingBTPCHW_shape).to(device)		# 'spacingDataX' would have already been cloned if cloneTensors == True.  No need to clone again.
	spacingY_BTPCHW = spacingDataY.view(spacingBTPCHW_shape).to(device)		# 		Same for spacingDataY.


	# I'm assuming that broadcastability is an equivalence relation.  If that is the case, then one can infer whether or not all
	# these tensors mutually broadcastable with only three checks.
	if not (
				(check_tensors_broadcastable(fieldBTPCHW, wavelengthsBTPCHW)) and
				(check_tensors_broadcastable(fieldBTPCHW, spacingX_BTPCHW)) and
				(check_tensors_broadcastable(fieldBTPCHW, spacingY_BTPCHW))
			):
		raise Exception("Incompatible dimensions encountered.")


	for dimInd in range(6):
		if (dimInd == 0):
			inds = batch_inds_range
		elif (dimInd == 1):
			inds = time_inds_range
		elif (dimInd == 2):
			inds = pupil_inds_range
		elif (dimInd == 3):
			inds = channel_inds_range
		elif (dimInd == 4):
			inds = height_inds_range
		elif (dimInd == 5):
			inds = width_inds_range

		curMaxDimSize = maxDim_shape[dimInd]
		curDimSizeField = fieldBTPCHW_shape[dimInd]
		curDimSizeWavelengths = wavelengthsBTPCHW_shape[dimInd]
		curDimSizeSpacing = spacingBTPCHW_shape[dimInd]

		inds_field = getDimIndicesList(inds, curDimSizeField, curMaxDimSize).to(device=device)
		inds_wavelengths = getDimIndicesList(inds, curDimSizeWavelengths, curMaxDimSize).to(device=device)
		inds_spacing = getDimIndicesList(inds, curDimSizeSpacing, curMaxDimSize).to(device=device)

		fieldBTPCHW = torch.index_select(fieldBTPCHW, dimInd, inds_field)
		wavelengthsBTPCHW = torch.index_select(wavelengthsBTPCHW, dimInd, inds_wavelengths)
		spacingX_BTPCHW = torch.index_select(spacingX_BTPCHW, dimInd, inds_spacing)
		spacingY_BTPCHW = torch.index_select(spacingY_BTPCHW, dimInd, inds_spacing)


	# Putting the spacing data tensor back together
	spacingBTPCHW = torch.cat((spacingX_BTPCHW, spacingY_BTPCHW), 4)

	# Converting the spacing from BTPCHW dimensions to TCD dimensions
	#	This is being done because it seems like parts of the Holotorch library code---e.g. parts of ASM_Prop.py---assume TCD dimensions
	#	for the SpacingContainer dimensions.
	spacingTCD_dim = Dimensions.TCD(n_time=spacingBTPCHW.shape[1], n_channel=spacingBTPCHW.shape[3], height=spacingBTPCHW.shape[4])
	spacingTCD = spacingBTPCHW.view(spacingTCD_dim.shape)
	

	new_wavelength_container =	WavelengthContainer(	wavelengths = wavelengthsBTPCHW,
														tensor_dimension = Dimensions.BTPCHW(	n_batch = wavelengthsBTPCHW.shape[0],
																								n_time = wavelengthsBTPCHW.shape[1],
																								n_pupil = wavelengthsBTPCHW.shape[2],
																								n_channel = wavelengthsBTPCHW.shape[3],
																								height = wavelengthsBTPCHW.shape[4],
																								width = wavelengthsBTPCHW.shape[5],
																							)
													)
	new_wavelength_container = new_wavelength_container.to(device=device)

	new_spacing_container =	SpacingContainer(spacing = spacingTCD, tensor_dimension = spacingTCD_dim)
	new_spacing_container = new_spacing_container.to(device=device)
	new_spacing_container.set_spacing_center_wavelengths(new_spacing_container.data_tensor)

	newField = 	ElectricField(
					data = fieldBTPCHW,
					wavelengths = new_wavelength_container,
					spacing = new_spacing_container
				)

	return newField