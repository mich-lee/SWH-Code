import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

import pathlib
import copy

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

from holotorch_addons.HelperFunctions import computeSpatialFrequencyGrids, computeBandlimitASM, computeBandlimitingFilterSpaceDomain, computeBandlimitingFilterASM, get_field_slice, print_cuda_memory_usage, fit_image_to_resolution
from holotorch_addons.Field_Resampler import Field_Resampler
from holotorch_addons.SimpleDetector import SimpleDetector
from holotorch_addons.Thin_Lens import Thin_Lens

import holotorch_addons.HolotorchPatches as HolotorchPatches
from Synthetic_Data_Generator import Synthetic_Data_Generator
from ASM_Prop_Patched import ASM_Prop_Patched
from Field_Padder_Unpadder import Field_Padder_Unpadder

warnings.filterwarnings('always',category=UserWarning)

################################################################################################################################

#### CUDA SETTINGS ####
use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

#### FOCUSING LENS SETTINGS ####
objectPlaneDistance = 150*mm #250*mm #50 * cm
imagePlaneDistance = 150*mm #107.1428571*mm #88.23529414 * mm
lensFocalLength = 75 * mm

wavelen = [854*nm, 856*nm]
res1 = (1518,2012)
delta1 = 1.85*um

inputImage = fit_image_to_resolution(Image.fromarray(imageio.imread('northwestern.png')), res1)
# inputImage.show()
inputImageDepthRange = 1*mm

inputImageData = torch.tensor(np.asarray(inputImage))
# inputImageData = (inputImageData[:,:,0] / 255) * (inputImageData[:,:,1] / 255) * torch.exp(1j * (2*np.pi/wavelen) * inputImageDepthRange * (inputImageData[:,:,0] / 255))
inputImageData = torch.exp(1j * (2*np.pi/torch.tensor(wavelen)[:,None,None]) * inputImageDepthRange * (inputImageData[:,:,0] / 255)) * ((inputImageData[:,:,1] / 255) > 0.1)

wavelenContainer = WavelengthContainer(wavelengths=wavelen, tensor_dimension=Dimensions.C(n_channel=np.size(wavelen)))
fieldIn = ElectricField.ones(size=[1,1,1,len(wavelen)]+list(res1), wavelengths=wavelenContainer, spacing=delta1)
fieldIn.data = fieldIn.data.to(torch.cfloat)
fieldIn.data[...,:,:] = inputImageData
fieldIn.data = fieldIn.data.to(device)
fieldIn.wavelengths = fieldIn.wavelengths.to(device)
fieldIn.spacing = fieldIn.spacing.to(device)


def propTest(inputField, asm, lens, unpadder = None):
	if (unpadder is None):
		return asm(lens(asm(inputField)))
	else:
		return unpadder(asm(lens(asm(inputField))))


fieldPadder1 = Field_Padder_Unpadder(pad_x = 759, pad_y = 1006)
lens1 = Thin_Lens(focal_length = lensFocalLength)
asmList = []
fieldUnpadder1 = Field_Padder_Unpadder(pad_x = -759, pad_y = -1006)


# Testing for bugs:
	# asmList.append(ASM_Prop_Patched(init_distance = 150*mm, padding_scale = 0, prop_computation_type='TF', bandlimit_kernel=False))
	# asmList.append(ASM_Prop_Patched(init_distance = 150*mm, padding_scale = 0, prop_computation_type='TF', prop_kernel_type=ENUM_PROP_KERNEL_TYPE.FULL_KERNEL, bandlimit_kernel=True))
	# asmList.append(ASM_Prop_Patched(init_distance = 150*mm, padding_scale = 0, prop_computation_type='TF', prop_kernel_type=ENUM_PROP_KERNEL_TYPE.PARAXIAL_KERNEL, bandlimit_kernel=True))
	# asmList.append(ASM_Prop_Patched(init_distance = 150*mm, padding_scale = 0, prop_computation_type='IR', prop_kernel_type=ENUM_PROP_KERNEL_TYPE.FULL_KERNEL, bandlimit_kernel=True))
	# asmList.append(ASM_Prop_Patched(init_distance = 150*mm, padding_scale = 0, prop_computation_type='IR', prop_kernel_type=ENUM_PROP_KERNEL_TYPE.PARAXIAL_KERNEL, bandlimit_kernel=True))



paddedFieldIn = fieldPadder1(fieldIn)
fieldOutputs = [paddedFieldIn]
for i in range(len(asmList)):
	fieldOutputs.append(propTest(paddedFieldIn, asmList[i], lens1, fieldUnpadder1))


plot_channel_num = 0
numCols = 3
plt.figure()
for i in range(len(fieldOutputs)):
	plt.figure((i // numCols) + 1)
	plt.subplot(2, numCols, (i % numCols) + 1)
	get_field_slice(fieldOutputs[i], channel_inds_range=plot_channel_num, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis = True)
	plt.subplot(2, numCols, (i % numCols) + 1 + numCols)
	get_field_slice(fieldOutputs[i], channel_inds_range=plot_channel_num, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis = True)


# plt.subplot(2,4,1)
# get_field_slice(o1, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis = True)
# plt.subplot(2,4,5)
# get_field_slice(o1, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis = True)
# plt.subplot(2,4,2)
# get_field_slice(o2, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis = True)
# plt.subplot(2,4,6)
# get_field_slice(o2, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis = True)
# plt.subplot(2,4,3)
# get_field_slice(o3, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis = True)
# plt.subplot(2,4,7)
# get_field_slice(o3, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis = True)
# plt.subplot(2,4,4)
# get_field_slice(o4, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis = True)
# plt.subplot(2,4,8)
# get_field_slice(o4, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis = True)

fieldIn = fieldIn