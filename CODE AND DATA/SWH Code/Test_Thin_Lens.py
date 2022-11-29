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

# S_obj = S_image = 150mm at 3036x4024 with 1.85um pixel pitch worked decently for 'northwestern.png'

#### FOCUSING LENS SETTINGS ####
objectPlaneDistance = 150*mm #250*mm #50 * cm
imagePlaneDistance = 150*mm #107.1428571*mm #88.23529414 * mm
lensFocalLength = 75 * mm

wavelen = [854*nm]
res1 = (1518,2012)
delta1 = 1.85*um
res2 = (1518,2012)
delta2 = 1.85*um
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


fieldPadder1 = Field_Padder_Unpadder(pad_x = 759, pad_y = 1006)
asm1 = ASM_Prop_Patched(init_distance = objectPlaneDistance, padding_scale = 0, prop_computation_type='TF', prop_kernel_type=ENUM_PROP_KERNEL_TYPE.FULL_KERNEL)
lens1 = Thin_Lens(focal_length = lensFocalLength)
asm2 = ASM_Prop_Patched(init_distance = imagePlaneDistance, padding_scale = 0, prop_computation_type='TF', prop_kernel_type=ENUM_PROP_KERNEL_TYPE.FULL_KERNEL)
fieldUnpadder1 = Field_Padder_Unpadder(pad_x = -759, pad_y = -1006)
resampler1 = Field_Resampler(	outputHeight = res2[0],
								outputWidth = res2[1],
								outputPixel_dx = delta2,
								outputPixel_dy = delta2,
								device = device
							)

model = torch.nn.Sequential(fieldPadder1, asm1, lens1, asm2, fieldUnpadder1, resampler1)

fieldPadder1.add_output_hook()
asm1.add_output_hook()
lens1.add_output_hook()
asm2.add_output_hook()
resampler1.add_output_hook()

fieldOut = model(fieldIn)


plt.figure()
plt.clf()
plt.subplot(2,4,1)
fieldPadder1.outputs[-1].visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis = True)
plt.subplot(2,4,5)
fieldPadder1.outputs[-1].visualize(plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis = True)
plt.subplot(2,4,2)
asm1.outputs[-1].visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis = True)
plt.subplot(2,4,6)
asm1.outputs[-1].visualize(plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis = True)
plt.subplot(2,4,3)
asm2.outputs[-1].visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis = True)
plt.subplot(2,4,7)
asm2.outputs[-1].visualize(plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis = True)
plt.subplot(2,4,4)
resampler1.outputs[-1].visualize(plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis = True)
plt.subplot(2,4,8)
resampler1.outputs[-1].visualize(plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis = True)

fieldIn = fieldIn








# Testing stuff with simulating lenses
#	The lens stuff seems alright (e.g. can focus as expected), but there appear to be some computational issues.
#	It seems like making the grid point spacing too large messes with the results.
#	However, making the grid spacing too small either results in the image on the image plane being too small, or the tensors being too large.
# tempModel = torch.nn.Sequential(
# 									ASM_Prop_Patched(init_distance = objectPlaneDistance, padding_scale = 1.5, utilize_cpu = False),
# 									# ASM_Prop(init_distance = -objectPlaneDistance),
# 									Thin_Lens(focal_length = lensFocalLength),
# 									ASM_Prop_Patched(init_distance = imagePlaneDistance, padding_scale = 1.5, utilize_cpu = False),
# 									slmResToSensorResResampler
# 								)
# subsamplingMagnitude = 1
# # tempFieldData = copy.deepcopy(synthDataGenerator.imagePlaneField.data)
# wasdf = synthDataGenerator.imagePlaneField.data.angle()
# wasdf[synthDataGenerator.imagePlaneField.data.abs() < 0.01] = 0
# B,T,P,C,H,W = wasdf.shape
# wasdf = wasdf.view(B*T*P,C,H,W)
# tempFieldData = torch.nn.functional.interpolate(wasdf, size=slmPlaneResolution, mode='bicubic')
# tempFieldData = tempFieldData.view(B,T,P,C,slmPlaneResolution[0], slmPlaneResolution[1])
# # tempFieldData[synthDataGenerator.imagePlaneField.data.abs() < 0.01] = 0
# tempFieldData = tempFieldData[... , 0::subsamplingMagnitude, 0::subsamplingMagnitude]
# tempField = ElectricField(tempFieldData, synthDataGenerator.wavelengths, 5*um)
# # tempField = ElectricField(tempFieldData, synthDataGenerator.wavelengths, subsamplingMagnitude*(1 / abs(synthDataGenerator.calc_magnification(objectPlaneDistance)))*sensorPlanePixelPitch)
# tempField.spacing.to(device)
# tempOutput = tempModel(tempField)
# plt.clf()
# get_field_slice(tempOutput, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize()