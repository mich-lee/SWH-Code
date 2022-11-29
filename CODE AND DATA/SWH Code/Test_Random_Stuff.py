import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

import pathlib
import copy
import re

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
# from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
from holotorch.utils.Enumerators import *
# from holotorch.Optical_Components.Resize_Field import Resize_Field
# from holotorch.Sensors.Detector import Detector
from holotorch.Optical_Components.Optical_Aperture import Optical_Aperture
# from holotorch.Optical_Components.Radial_Optical_Aperture import Radial_Optical_Aperture

from holotorch_addons.HelperFunctions import computeSpatialFrequencyGrids, computeBandlimitASM, computeBandlimitingFilterSpaceDomain, computeBandlimitingFilterASM, get_field_slice, print_cuda_memory_usage, fit_image_to_resolution, parseNumberAndUnitsString
# from holotorch_addons.Field_Resampler import Field_Resampler
# from holotorch_addons.SimpleDetector import SimpleDetector
from holotorch_addons.Thin_Lens import Thin_Lens
from holotorch_addons.Radial_Optical_Aperture_Patched import Radial_Optical_Aperture_Patched

import holotorch_addons.HolotorchPatches as HolotorchPatches
# from Synthetic_Data_Generator import Synthetic_Data_Generator
# from ASM_Prop_Patched import ASM_Prop_Patched
# from Field_Padder_Unpadder import Field_Padder_Unpadder
# from Fresnel_Two_Step_Prop import Fresnel_Two_Step_Prop

################################################################################################################################

#### CUDA SETTINGS ####
use_cuda = True
gpu_no = 0
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

################################################################################################################################


# fieldIn = ElectricField.ones(size=[1,1,1,1,8000,8000], wavelengths=854*nm, spacing=5*um)
# fieldIn.data = fieldIn.data.to(torch.cfloat)
# fieldIn.data = fieldIn.data.to(device)
# fieldIn.wavelengths = fieldIn.wavelengths.to(device)
# fieldIn.spacing = fieldIn.spacing.to(device)

# aperture = Radial_Optical_Aperture_Patched(aperture_radius=50.8/2*mm, off_x=0*mm, off_y=0*mm)

# fieldOut = aperture(fieldIn)

# fieldOut.visualize(flag_axis = True)


asdf = parseNumberAndUnitsString('-.76um')

strs = ['12', '123mm', '.12ns', '-.76um','123m','-30cm','1203.','4534.us']
for s in strs:
	print('%g, %s, %s' % parseNumberAndUnitsString(s))

# def getSuffixStr(str):
# 	underscoreIndex = None
# 	for match in re.finditer('_', str):
# 		underscoreIndex = match.start()
	
# 	if underscoreIndex is None:
# 		return None
	
# 	suffixStrTemp = str[underscoreIndex+1:]
# 	if len(suffixStrTemp) == 0:
# 		return None

# 	dotIndex = None
# 	for match in re.finditer('\.', suffixStrTemp):
# 		dotIndex = match.start()

# 	if dotIndex is None:
# 		return suffixStrTemp
# 	suffixStrTemp2 = suffixStrTemp[0:dotIndex]
# 	if len(suffixStrTemp2) == 0:
# 		return None
# 	return suffixStrTemp2


# strs = ['asdf_qwerty_.png', 'dfassdfd_hello', 'xgdfsfgas_goodbye.avi', 'Fourier', 'stanford_bunny.png', 'stanford_bunny_.124.23mm.png']
# for s in strs:
# 	print('%s' % (getSuffixStr(s)))






1 == 2