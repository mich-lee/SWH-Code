########################################################################################################################
####    WISHED Experimental Setup Notes                                                                             ####
########################################################################################################################
# PAPER: "WISHED: Wavefront imaging sensor with high resolution and depth ranging" (Wu and Li et al, 2020)
# EQUIPMENT:
#	SPATIAL LIGHT MODULATOR:
#		- MODEL: HOLOEYE LETO (probably the LETO-3-NIR-081)
#		- RESOLUTION: 1920x1080 pixels, 6.4um pitch size			(Pixel resolution is given as width x height)
#		- MAXIMUM PHASE SHIFT:
#			- 4.4*pi @ 650 nm										(Given in specs)
#			- 2.4*pi @ 1064 nm										(Given in specs)
#			- 3.414492756*pi @ 854nm								(Interpolated from given data)
#	FOCUSING LENS
#		- MODEL: Thorlabs AC508-075-B-ML
#		- FOCAL LENGTH: 75.0mm
#		- LENS DIAMETER: 50.8mm
#	IMAGE SENSOR (CAMERA)
#		- NAME: Basler Ace
# 		- MODEL NUMBER: acA4024-29um
#		- RESOLUTION: 4024x3036 pixels, 1.85x1.85 um pitch size		(Pixel resolution is given as width x height)
#		- Bit depth: 10 bits 										(Stated in the paper)
#	OTHER EQUIPMENT
#		- TUNEABLE LASER
#			- MODEL: Toptica DLC pro850
#			- CENTER WAVELENGTH: 850nm
#		- COLLIMATING LENS
#		- LINEAR POLARIZER
# SETUP:
#		- Focusing lens located at z = 0cm
#		- Object placed approximately 50cm in front of the focusing lens, i.e. at approximately z = -50cm
#			- Note that points on the object's surface generally will not all lie in the z = -50cm plane as
#			  the object is 3D.
#				- The paper claimed unambiguous depth ranges of up to 1.2cm, so one can probably assume that
#				  the variation in in z will be less than or equal to 1.2cm.
#						- 1.2cm is relatively small compared to 50cm.
#			- Because the lens is not infinitely thin, the 50cm distance is ambiguous.  The Thorlabs website
#			  indicates that the lens (Thorlabs AC508-075-B-ML) is about 19.8mm in length along the axial direction.
#			  However, 19.8mm is somewhat small compared to 50cm.
#		- The sensor should be placed in the image plane of the focusing lens
#			- As far as I can tell, it was not explicitly stated that the sensor was placed in the image plane.
#			  However, it was explicitly indicated that the sensor was placed in the image plane in the simulations.
#			  Additionally, it would make sense to have a focused image on either the sensor or the SLM (e.g. it would
#			  make sense to have the sensor be in the image plane) as the 
#			  
#	
#
#
#
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
#	- WISHED paper λ(nm): 854.31, 854.43, 854.71, 855.73
############################################################################








#### BEGIN BACKUP REGION ####

########################################################################################################################
## ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ##
## █            ▄▄▄▄▄▄▄▄▄    ▄▄     ▄▄    ▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄             █ ##
## █           ▐░░░░░░░░░▌  ▐░░▌   ▐░░▌  ▐░░░░░░░░░▌  ▐░░░░░░░░░▌  ▐░░░░░░░░░▌  ▐░░░░░░░░░▌  ▐░░░░░░░░░▌            █ ##
## █            ▀▀▀█░█▀▀▀   ▐░▌░▌ ▐░▐░▌  ▐░█▀▀▀▀▀█░▌  ▐░█▀▀▀▀▀█░▌  ▐░█▀▀▀▀▀█░▌   ▀▀▀█░█▀▀▀   ▐░█▀▀▀▀▀▀▀             █ ##
## █               ▐░▌      ▐░▌▐░▐░▌▐░▌  ▐░█▄▄▄▄▄█░▌  ▐░▌     ▐░▌  ▐░█▄▄▄▄▄█░▌      ▐░▌      ▐░█▄▄▄▄▄▄▄             █ ##
## █               ▐░▌      ▐░▌ ▐░▌ ▐░▌  ▐░░░░░░░░░▌  ▐░▌     ▐░▌  ▐░░░░░░░░░▌      ▐░▌      ▐░░░░░░░░░▌            █ ##
## █               ▐░▌      ▐░▌  ▀  ▐░▌  ▐░█▀▀▀▀▀▀▀   ▐░▌     ▐░▌  ▐░█▀▀▀█░█▀       ▐░▌       ▀▀▀▀▀▀▀█░▌            █ ##
## █            ▄▄▄█░█▄▄▄   ▐░▌     ▐░▌  ▐░▌          ▐░█▄▄▄▄▄█░▌  ▐░▌    ▐░▌       ▐░▌       ▄▄▄▄▄▄▄█░▌            █ ##
## █           ▐░░░░░░░░░▌  ▐░▌     ▐░▌  ▐░▌          ▐░░░░░░░░░▌  ▐░▌     ▐░▌      ▐░▌      ▐░░░░░░░░░▌            █ ##
## █            ▀▀▀▀▀▀▀▀▀    ▀       ▀    ▀            ▀▀▀▀▀▀▀▀▀    ▀       ▀        ▀        ▀▀▀▀▀▀▀▀▀             █ ##
## ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀ ##
########################################################################################################################
from turtle import clone
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

from holotorch_addons.HelperFunctions import computeSpatialFrequencyGrids, computeBandlimitASM, computeBandlimitingFilterSpaceDomain, computeBandlimitingFilterASM, get_field_slice, print_cuda_memory_usage
from holotorch_addons.Field_Resampler import Field_Resampler
from holotorch_addons.SimpleDetector import SimpleDetector
from holotorch_addons.Thin_Lens import Thin_Lens

import holotorch_addons.HolotorchPatches as HolotorchPatches
from Synthetic_Data_Generator import Synthetic_Data_Generator
from ASM_Prop_Patched import ASM_Prop_Patched

from holotorch_addons.Thin_Lens import Thin_Lens
from holotorch_addons.Radial_Optical_Aperture_Patched import Radial_Optical_Aperture_Patched
from ASM_Prop_Patched import ASM_Prop_Patched
# from Field_Padder_Unpadder import Field_Padder_Unpadder
from Fresnel_Two_Step_Prop import Fresnel_Two_Step_Prop
from Multi_Resolution_Sequential import Multi_Resolution_Sequential

warnings.filterwarnings('always',category=UserWarning)

################################################################################################################################
















########################################################################################################################
## ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ##
## █      ▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄    ▄▄      ▄    ▄▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄▄      █ ##
## █     ▐░░░░░░░░░▌  ▐░░░░░░░░░▌  ▐░░░░░░░░░▌  ▐░░░░░░░░░▌  ▐░░░░░░░░░▌  ▐░░▌    ▐░▌  ▐░░░░░░░░░▌  ▐░░░░░░░░░▌     █ ##
## █     ▐░█▀▀▀▀▀▀▀   ▐░█▀▀▀▀▀▀▀    ▀▀▀█░█▀▀▀    ▀▀▀█░█▀▀▀    ▀▀▀█░█▀▀▀   ▐░▌░▌   ▐░▌  ▐░█▀▀▀▀▀▀▀   ▐░█▀▀▀▀▀▀▀      █ ##
## █     ▐░█▄▄▄▄▄▄▄   ▐░█▄▄▄▄▄▄▄       ▐░▌          ▐░▌          ▐░▌      ▐░▌▐░▌  ▐░▌  ▐░▌ ▄▄▄▄▄▄   ▐░█▄▄▄▄▄▄▄      █ ##
## █     ▐░░░░░░░░░▌  ▐░░░░░░░░░▌      ▐░▌          ▐░▌          ▐░▌      ▐░▌ ▐░▌ ▐░▌  ▐░▌▐░░░░░░▌  ▐░░░░░░░░░▌     █ ##
## █      ▀▀▀▀▀▀▀█░▌  ▐░█▀▀▀▀▀▀▀       ▐░▌          ▐░▌          ▐░▌      ▐░▌  ▐░▌▐░▌  ▐░▌ ▀▀▀▀█░▌   ▀▀▀▀▀▀▀█░▌     █ ##
## █      ▄▄▄▄▄▄▄█░▌  ▐░█▄▄▄▄▄▄▄       ▐░▌          ▐░▌       ▄▄▄█░█▄▄▄   ▐░▌   ▐░▐░▌  ▐░█▄▄▄▄▄█░▌   ▄▄▄▄▄▄▄█░▌     █ ##
## █     ▐░░░░░░░░░▌  ▐░░░░░░░░░▌      ▐░▌          ▐░▌      ▐░░░░░░░░░▌  ▐░▌    ▐░░▌  ▐░░░░░░░░░▌  ▐░░░░░░░░░▌     █ ##
## █      ▀▀▀▀▀▀▀▀▀    ▀▀▀▀▀▀▀▀▀        ▀            ▀        ▀▀▀▀▀▀▀▀▀    ▀      ▀▀    ▀▀▀▀▀▀▀▀▀    ▀▀▀▀▀▀▀▀▀      █ ##
## ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀ ##
########################################################################################################################


########################################################################################################################
####    OTHER SETTINGS                                                                                              ####
########################################################################################################################
#### SENSOR AND SLM PLANE RESOLUTION SETTINGS ####
slmPlaneResolution = (270*4, 480*4)			# Size of SLM plane in pixels; (height,width)
slmPlanePixelPitch = 6.4*um					# Spacing between pixels in the sensor plane

#### SLM SETTINGS ####
sensorPlaneResolution = (759*4, 1006*4) 		# Size of sensor plane in pixels; (height,width)
sensorPlanePixelPitch = 1.85*um 			# Spacing between pixels in the sensor plane
slmToSensorPlaneDistance = 25 * mm
slmMaxPhaseShift = 3.414492756 * np.pi

#### IMAGE PLANE SETTINGS ####
imagePlaneLocation = 'sensor'

#### FOCUSING LENS AND IMAGING SETTINGS ####
objectPlaneDistance = 50 * cm
imagePlaneDistance = 88.23529414 * mm
lensFocalLength = 75 * mm
objectDistanceDelta = 0 * mm

#### WAVELENGTH AND SPECKLE SETTINGS ####
wavelengthsToProcess = [854.31*nm, 854.43*nm, 854.71*nm, 855.73*nm]
numSpecklePatternsPerWavelength = 24



########################################################################################################################
####    DATA FOLDER SETTINGS                                                                                        ####
########################################################################################################################
# NOTES:
#	The directory structure will be as follows:
#		data_folder_root_str/
#		 |	temp_slm_data_folder_str/
#			 |	...temporary SLM data files...
#		 |	save_data_directory_str/
#			 |	...folders containing generated synthetic data...
data_folder_root_str = "../DATA"
temp_slm_data_folder_str = ".temp_slm"
save_data_directory_str = None # "Synthetic Data Temp"



########################################################################################################################
####    3D DATA INPUT SETTINGS                                                                                      ####
########################################################################################################################
# NOTES:
#	- The code loads in the image specified by 'inputImageFilename' and extracts the RGBA channels.
#	- This image specifies the 3D object being images
#		- The red channel is used to determine the depth, which is scaled to span 'inputImageDepthRange'
#		- The green channel controls how much each point reflects, with [0,255] being mapped linearly to [0,1]
inputImageFilename = 'northwestern.png'
objectDepthRange = 1 * mm



########################################################################################################################
####    COMPUTATION SETTINGS                                                                                        ####
########################################################################################################################
#### IMAGE PLANE CALCULATION SETTINGS ####
imagePlaneCalculationType = 'propagation_1'
## Settings for imagePlaneCalculationType = 'propagation_1' ##	<--- Settings will be set to None if imagePlaneCalculationType != 'propagation_1'
objectPlaneResolution = (4000,4000)
objectPlaneSampleSpacing = 2.5*um
desiredObjectDimensions = (10*mm, 10*mm)

#### SLM COMPUTATION SETTINGS ####
numSlmMiniBatches = 6
store_slm_temp_data_on_disk = False

#### CUDA SETTINGS ####
use_cuda = True
gpu_no = 0
















#################################################################
# Miscellaneous setup
#################################################################
wavelengthsArray = wavelengthsToProcess
numChannels = len(wavelengthsArray)

if (imagePlaneLocation == 'sensor'):
	lensToSLMPlaneDistance = imagePlaneDistance - slmToSensorPlaneDistance
else:
	lensToSLMPlaneDistance = imagePlaneDistance

# wavelengths = WavelengthContainer(
# 			wavelengths = wavelengthsArray,
# 			tensor_dimension = Dimensions.C(n_channel=numChannels)
# )
# wavelengths = wavelengths.to(device)

data_folder_root = pathlib.Path(data_folder_root_str)
temp_slm_data_folder = data_folder_root / temp_slm_data_folder_str
if (save_data_directory_str is None):
	save_data_directory = None
else:
	save_data_directory = data_folder_root / save_data_directory_str

# print(data_folder_root.resolve())
# print(temp_slm_data_folder.resolve())
# print(slm_data_folder.resolve())

if (imagePlaneCalculationType != 'propagation_1'):
	objectPlaneResolution = None
	objectPlaneSampleSpacing = None

# Setting device
device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")

# For saving additional information along with the SLM and sensor plane data
# Use this to provide more information about the setup
extraMetadataDictionary =	{
								'slmToSensorPlaneDistance'	: slmToSensorPlaneDistance,
							}
















#################################################################
# Doing stuff
#################################################################
slmModel =	HolotorchPatches.SLM_PhaseOnly_Patched.create_slm(	height          = slmPlaneResolution[0],
																width           = slmPlaneResolution[1],
																n_channel       = numChannels,
																n_batch			= numSpecklePatternsPerWavelength,
																n_slm_batches	= numSlmMiniBatches,
																feature_size    = slmPlanePixelPitch,
																init_type       = ENUM_SLM_INIT.RANDOM,
																init_variance   = slmMaxPhaseShift,
																slm_directory	= temp_slm_data_folder.resolve(),
																store_on_gpu	= not store_slm_temp_data_on_disk,
																device			= device
															)

slmToSensorPlanePropASM = ASM_Prop_Patched(init_distance = slmToSensorPlaneDistance, sign_convention = HolotorchPatches.ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE)
slmResToSensorResResampler = 	Field_Resampler(
									outputHeight = sensorPlaneResolution[0],
									outputWidth = sensorPlaneResolution[1],
									outputPixel_dx = sensorPlanePixelPitch,
									outputPixel_dy = sensorPlanePixelPitch,
									device = device
								)
slmToSensorPlanePropModel = torch.nn.Sequential(slmToSensorPlanePropASM, slmResToSensorResResampler)

detectorModel = SimpleDetector()

slmInputBandlimitingFilter = computeBandlimitingFilterASM(wavelengthsArray, slmPlaneResolution, slmPlanePixelPitch, lensToSLMPlaneDistance, device=device)





prop1 = ASM_Prop_Patched(init_distance=objectPlaneDistance, do_padding=True, do_unpad_after_pad=False)
# prop1 = Fresnel_Two_Step_Prop(M=8000, delta1=2.5*um, delta2=2.5*um, propagationDistance=objectPlaneDistance, resampleAtInput=False, device=device)
lens_stop = Radial_Optical_Aperture_Patched(aperture_radius=50.8*mm/2)
lens1 = Thin_Lens(focal_length = lensFocalLength)
prop2 = Fresnel_Two_Step_Prop(M=8000, delta1=2.5*um, delta2=sensorPlanePixelPitch, propagationDistance=imagePlaneDistance, resampleAtInput=False, device=device)
# prop2 = ASM_Prop_Patched(init_distance=imagePlaneDistance, do_padding=False)

multiresComponents 	= [							prop1,						lens_stop,				lens1,				prop2														]
multiresResolutions	= [		(4000, 4000),				(), 						(),					(),					sensorPlaneResolution							]
multiresSpacings	= [		(2.5*um, 2.5*um),			(),					(),					(),					(sensorPlanePixelPitch,sensorPlanePixelPitch)	]
# multiresResolutions	= [		(4000, 4000),				(8000, 8000), 						(),					(),					sensorPlaneResolution							]
# multiresSpacings	= [		(2.5*um, 2.5*um),			(2.5*um, 2.5*um),					(),					(),					(sensorPlanePixelPitch,sensorPlanePixelPitch)	]

objectPlaneToImagePlaneModel = Multi_Resolution_Sequential(multiresComponents, multiresResolutions, multiresSpacings, device=device)









synthDataGenerator = Synthetic_Data_Generator(
	save_data_directory = save_data_directory,
	creator_script_path = sys.argv[0],
	######################################################
	sensorPlaneResolution = sensorPlaneResolution,
	sensorPlanePixelPitch = sensorPlanePixelPitch,
	slmPlaneResolution = slmPlaneResolution,
	slmPlanePixelPitch = slmPlanePixelPitch,
	######################################################
	wavelengths = wavelengthsArray,
	######################################################
	inputImageFilepath = inputImageFilename,
	objectDepthRange = objectDepthRange,
	######################################################
	imagePlaneType = imagePlaneLocation,
	######################################################
	imagePlaneCalculationType = imagePlaneCalculationType,
	focusingLensFocalLength = lensFocalLength,
	objectPlaneDistance = objectPlaneDistance,
	imagePlaneDistance = imagePlaneDistance,
	objectDistanceDelta = objectDistanceDelta,
	######################################################
	objectPlaneResolution = objectPlaneResolution,
	objectPlaneSampleSpacing = objectPlaneSampleSpacing,
	desiredObjectDimensions = desiredObjectDimensions,
	objectPlaneToImagePlaneModel = objectPlaneToImagePlaneModel,
	######################################################
	slmToSensorPlanePropModel = slmToSensorPlanePropModel,
	slmInputBandlimitingFilter = slmInputBandlimitingFilter,
	slmModel = slmModel,
	detectorModel = detectorModel,
	######################################################
	saveSensorMeasurementsInMemory = True,
	useFloat16ForIntensity = True,
	extraMetadataDictionary = extraMetadataDictionary,
	device = device,
	######################################################
	_debug_skip_long_ops = False
)




#### END BACKUP REGION ####




w = objectPlaneToImagePlaneModel.model
temp_field = get_field_slice(synthDataGenerator.objectPlaneField, channel_inds_range=0, field_data_tensor_dimension=Dimensions.BTPCHW, cloneTensors=True)

magnitudePlotCmapName = 'turbo'
phasePlotCmapName = 'twilight'

plt.figure(1)
plt.clf()

plt.subplot(2,4,1)
temp_field.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis=True)
plt.title('Object Plane Field\n(Magnitude)', {'fontweight':'bold','fontsize':14})
plt.set_cmap(magnitudePlotCmapName)
plt.subplot(2,4,5)
temp_field.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis=True)
plt.title('Object Plane Field\n(Phase)', {'fontweight':'bold','fontsize':14})
plt.set_cmap(phasePlotCmapName)
plt.clim(-np.pi, np.pi)

temp = w[0](temp_field)
# plt.subplot(2,4,2)
# temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis=True)
# plt.subplot(2,4,6)
# temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis=True)

temp = w[1](temp)
plt.subplot(2,4,2)
temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis=True)
plt.title('Field After ASM Propagation\n(Magnitude)', {'fontweight':'bold','fontsize':14})
plt.set_cmap(magnitudePlotCmapName)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.subplot(2,4,6)
temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis=True)
plt.title('Field After ASM Propagation\n(Phase)', {'fontweight':'bold','fontsize':14})
plt.set_cmap(phasePlotCmapName)
plt.clim(-np.pi, np.pi)
plt.xlim(-6, 6)
plt.ylim(-6, 6)

temp = w[2](temp)
# plt.subplot(2,4,4)
# temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis=True)
# plt.subplot(2,4,8)
# temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis=True)

# plt.figure(2)
# plt.clf()

temp = w[3](temp)
plt.subplot(2,4,3)
temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis=True)
plt.title('Field After Lens\n(Magnitude)', {'fontweight':'bold','fontsize':14})
plt.set_cmap(magnitudePlotCmapName)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.subplot(2,4,7)
temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis=True)
plt.title('Field After Lens\n(Phase)', {'fontweight':'bold','fontsize':14})
plt.set_cmap(phasePlotCmapName)
plt.clim(-np.pi, np.pi)
plt.xlim(-6, 6)
plt.ylim(-6, 6)

temp = w[4](temp)
# plt.subplot(2,4,1)
# temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis=True)
# plt.subplot(2,4,5)
# temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis=True)

temp = w[5](temp)
plt.subplot(2,4,4)
temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis=True)
plt.title('Field After Fresnel Two-Step Prop.\n(Magnitude)', {'fontweight':'bold','fontsize':14})
plt.set_cmap(magnitudePlotCmapName)
plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)
plt.subplot(2,4,8)
temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis=True)
plt.title('Field After Fresnel Two-Step Prop.\n(Phase)', {'fontweight':'bold','fontsize':14})
plt.set_cmap(phasePlotCmapName)
plt.clim(-np.pi, np.pi)
plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)

# temp = w[6](temp)
# plt.subplot(2,4,3)
# temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.MAGNITUDE, flag_axis=True)
# plt.subplot(2,4,7)
# temp.visualize(rescale_factor = 0.25, plot_type=ENUM_PLOT_TYPE.PHASE, flag_axis=True)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.4)
plt.show()




plt.figure(figsize=(10,10))
plot_channel = 0
plt.subplot(231)
synthDataGenerator.get_field_slice_channel(synthDataGenerator.slmInputPlaneField, plot_channel).visualize(rescale_factor = 0.25, flag_colorbar=True, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title('Recovered SLM plane field (Magnitude)')
plt.subplot(234)
synthDataGenerator.get_field_slice_channel(synthDataGenerator.slmInputPlaneField, plot_channel).visualize(rescale_factor = 0.25, flag_colorbar=True, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
plt.title('Recovered SLM plane field (Phase)')
plt.subplot(232)
synthDataGenerator.get_field_slice_channel(synthDataGenerator.sensorPlaneFieldEstimateFromSlmPlane, plot_channel).visualize(rescale_factor = 0.25, flag_colorbar=True, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title('Sensor plane field estimate (Magnitude)')
plt.subplot(235)
synthDataGenerator.get_field_slice_channel(synthDataGenerator.sensorPlaneFieldEstimateFromSlmPlane, plot_channel).visualize(rescale_factor = 0.25, flag_colorbar=True, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
plt.title('Sensor plane field estimate (Phase)')
plt.subplot(233)
synthDataGenerator.get_field_slice_channel(synthDataGenerator.imagePlaneField, plot_channel).visualize(rescale_factor = 0.25, flag_colorbar=True, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
plt.title('"Ground truth" sensor plane field (Magnitude)')
plt.subplot(236)
synthDataGenerator.get_field_slice_channel(synthDataGenerator.imagePlaneField, plot_channel).visualize(rescale_factor = 0.25, flag_colorbar=True, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
plt.title('"Ground truth" sensor plane field (Phase)')


pass