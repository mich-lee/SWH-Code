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
from holotorch_addons.Radial_Optical_Aperture_Patched import Radial_Optical_Aperture_Patched

import holotorch_addons.HolotorchPatches as HolotorchPatches
from Synthetic_Data_Generator import Synthetic_Data_Generator
from ASM_Prop_Patched import ASM_Prop_Patched
from Field_Padder_Unpadder import Field_Padder_Unpadder
from Fresnel_Two_Step_Prop import Fresnel_Two_Step_Prop
from Single_Lens_Focusing_System import Single_Lens_Focusing_System

################################################################################################################################

#### CUDA SETTINGS ####
use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

#### FOCUSING LENS SETTINGS ####
objectPlaneDistance = 500*mm #250*mm #50 * cm
imagePlaneDistance = 88.23529414*mm #107.1428571*mm #88.23529414 * mm
lensFocalLength = 75 * mm

wavelen = [854.31*nm, 854.43*nm] #854.43*nm #[854*nm, 856*nm]
res12 = (8000,8000)
delta1 = 5*um
delta2 = 1.85*um
res3 = (8000,8000)

inputImageDepthRange = 22.2815534*mm #1.4*mm
targetObjectSizeSpatial = (20*mm, 20*mm)

if (np.isscalar(wavelen)):
	wavelen = [wavelen]

targetImageSize = tuple((2*torch.round(torch.tensor(targetObjectSizeSpatial, dtype=torch.float) / delta1 / 2)).to(torch.int).tolist())
inputImage = fit_image_to_resolution(Image.fromarray(imageio.imread('northwestern.png')), targetImageSize)
# inputImage.show()
startInds = torch.floor((torch.tensor(res12) - torch.tensor(targetImageSize)) / 2)
endInds = startInds + torch.tensor(targetImageSize)
startInds = tuple(startInds.to(torch.int).tolist())
endInds = tuple(endInds.to(torch.int).tolist())


inputImageData = torch.tensor(np.asarray(inputImage))
# inputImageData = torch.exp(1j * (2*np.pi/wavelen) * inputImageDepthRange * (inputImageData[:,:,0] / 255)) * (inputImageData[:,:,1] / 255)
inputImageData = torch.exp(1j * (2*np.pi/torch.tensor(wavelen)[:,None,None]) * inputImageDepthRange * (inputImageData[:,:,0] / 255)) * (inputImageData[:,:,1] / 255)
	# inputImageData = (inputImageData[:,:,0] / 255) * (inputImageData[:,:,1] / 255) * torch.exp(1j * (2*np.pi/wavelen) * inputImageDepthRange * (inputImageData[:,:,0] / 255))

wavelenContainer = WavelengthContainer(wavelengths=wavelen, tensor_dimension=Dimensions.C(n_channel=np.size(wavelen)))
fieldIn = ElectricField.zeros(size=[1,1,1,len(wavelen)]+list(res12), wavelengths=wavelenContainer, spacing=delta1)
# fieldIn = ElectricField.zeros(size=[1,1,1,1]+list(res12), wavelengths=wavelenContainer, spacing=delta1)
fieldIn.data = fieldIn.data.to(torch.cfloat)
fieldIn.data[...,startInds[0]:endInds[0],startInds[1]:endInds[1]] = inputImageData
fieldIn.data = fieldIn.data.to(device)
fieldIn.wavelengths = fieldIn.wavelengths.to(device)
fieldIn.spacing = fieldIn.spacing.to(device)




# # fresnelTwo = Fresnel_Two_Step_Prop(M=8000, delta1=1.85*um, delta2=1.85*um, propagationDistance=88.23529414*mm, device=device)

# asm1 = ASM_Prop_Patched(init_distance=objectPlaneDistance, do_padding=False)
# # fresnelTwo1 = Fresnel_Two_Step_Prop(M=max(res12), delta1=delta1, delta2=delta1, propagationDistance=objectPlaneDistance, device=device)
# lens_stop = Radial_Optical_Aperture_Patched(aperture_radius=50.8/2*mm)
# lens1 = Thin_Lens(focal_length = lensFocalLength)
# fresnelTwo2 = Fresnel_Two_Step_Prop(M=max(res3), delta1=delta1, delta2=delta2, propagationDistance=imagePlaneDistance, device=device)

# # fieldOut1 = fresnelTwo1(fieldIn)
# fieldOut1 = asm1(fieldIn)
# fieldOut2 = lens_stop(fieldOut1)
# fieldOut3 = lens1(fieldOut2)
# fieldOut4 = fresnelTwo2(fieldOut3)


model = Single_Lens_Focusing_System(
			objectPlaneDistance=objectPlaneDistance,
			imagePlaneDistance=imagePlaneDistance,
			lensFocalLength=lensFocalLength,
			lensDiameter=50.8*mm,
			numSamplesPerSide=max(res12),
			inputPlaneSampleSpacing=delta1,
			outputPlaneSampleSpacing=delta2,
			device=device
		)

fieldOut4 = model(fieldIn)


synthField = fieldOut4.data[0,0,0,1,:,:] * fieldOut4.data[0,0,0,0,:,:].conj()
synthField = synthField[None,None,None,None,:,:]
wavelenContainer2 = WavelengthContainer(wavelengths=-1, tensor_dimension=Dimensions.C(n_channel=1))
fieldOutSynth = ElectricField(data=synthField, wavelengths=wavelenContainer, spacing=delta2)
fieldOutSynth.wavelengths = fieldOutSynth.wavelengths.to(device)
fieldOutSynth.spacing = fieldOutSynth.spacing.to(device)


numCols = 4
plt.figure()
plt.subplot(2,numCols,1)

get_field_slice(fieldIn, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.subplot(2,numCols,1+numCols)
get_field_slice(fieldIn, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)

plt.subplot(2,numCols,2)
get_field_slice(fieldOut4, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.ylim([-2.5,2.5])
plt.xlim([-2.5,2.5])
plt.subplot(2,numCols,2+numCols)
get_field_slice(fieldOut4, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
plt.ylim([-2.5,2.5])
plt.xlim([-2.5,2.5])

plt.subplot(2,numCols,3)
get_field_slice(fieldOut4, channel_inds_range=1, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.ylim([-2.5,2.5])
plt.xlim([-2.5,2.5])
plt.subplot(2,numCols,3+numCols)
get_field_slice(fieldOut4, channel_inds_range=1, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor=0.25, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
plt.ylim([-2.5,2.5])
plt.xlim([-2.5,2.5])

plt.subplot(2,numCols,4)
get_field_slice(fieldOutSynth, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.ylim([-2.5,2.5])
plt.xlim([-2.5,2.5])
plt.subplot(2,numCols,4+numCols)
get_field_slice(fieldOutSynth, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
plt.ylim([-2.5,2.5])
plt.xlim([-2.5,2.5])

1 == 2