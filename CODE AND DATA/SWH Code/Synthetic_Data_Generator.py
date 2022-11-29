from lib2to3.pytree import Base
# from logging import warning
from pathlib import Path
import string
from tkinter import E
from tkinter.messagebox import NO
import numpy as np
import sys
import os
import datetime
import torch
import matplotlib.pyplot as plt

from numpy import asarray
import re

# Image wranglers
import imageio
from PIL import Image

import warnings

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")
sys.path.append("holotorch_addons/")

# import holotorch
import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
# from holotorch.LightSources.CoherentSource import CoherentSource
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
from holotorch.utils.Enumerators import *
# from holotorch.Optical_Setups.Base_Setup import Base_Setup
from holotorch.Optical_Components.CGH_Component import CGH_Component

from holotorch_addons.Mesh_Functions import createMeshFromGridsAndDepths
from holotorch_addons.HelperFunctions import applyFilterToElectricField, get_field_slice, print_cuda_memory_usage, get_tensor_size_bytes, generateGrid, fit_image_to_resolution, parseNumberAndUnitsString
from holotorch_addons.Thin_Lens import Thin_Lens
# from Single_Lens_Focusing_System import Single_Lens_Focusing_System
from holotorch_addons.Field_Resampler import Field_Resampler

################################################################################################################################


################################################################################################################################
	############################################################################
	# Procedure:
	#	1. Determine/estimate u_{im}, i.e. the field at the image plane
	#		Methods (rough description):
	#			a) Load in image (more or less) directly onto image plane
	#			b) [NOT IMPLEMENTED]
	#					Load in an image and use that to specify a set of 3D points.
	#					Use ray optics to estimate field at image plane
	#						- Effects such as z-coordinate dependent magnification and image plane location
	#							mean that light will (generally) not be perfectly focused on a single image plane.
	#							This will reduce the accuracy of the ray optics approximation.
	#			c) Load in an image and use that to specify a set of 3D points.
	#				Apply a phase shift proportional to the z-coordinate offset (relative to object plane).
	#				Propagate from object plane to image plane.
	#					- This is an approximation.  It will probably be pretty inaccurate for small wavelengths and/or small transverse frequency components.
	#			d) [NOT IMPLEMENTED]
	#					Load in an image and use that to specify a set of 3D points.
	#					Use Huygens-Fresnel integral, but split up the integration into chunks to not run out of GPU memory.
	#					I.e. split xy-grids into smaller chunks.  E.g. if splitting into 8x8, do 8x8x8x8 vectorized operations for the integral.
	#	If the image plane is located at the SLM plane:		<-- Equivalently, imagePlaneType = 'slm'
	#		#### I.e. light is focused on the SLM plane ####
	#		If slmInputBandlimitingFilter != None:
	#			2. u_{slm,in} = Bandlimit(u_{im})
	#				a) u_{slm,in} is the field at the input of the SLM plane
	#				b) Bandlimiting is done by convolving u_{im} with slmInputBandlimitingFilter
	#		Else:
	#			2. u_{slm_in} = u_{im}
	#				a) u_{slm,in} is the field at the input of the SLM plane
	#		Skip #3 and go to #4
	#	Elseif the image plane is located at the sensor plane:		<-- Equivalently, imagePlaneType = 'sensor'
	#		#### I.e. light is focused on the sensor plane ####
	#		2. u_{sensor} = u_{im}
	#			a) u_{sensor} is the field at the sensor plane
	#		3. u_{slm_in} = Bandlimit(arg min (1 / # pixels) * \sum_{i}^{N} |u_{sensor} - Prop(Bandlimit(u'_{slm,in}))| ^ 2)
	#			a) This finds u'_{slm_in} that minimizes the mean-squared error between u_{sensor} and the field
	#				at the sensor plane that would result from propagating u'_{slm_in} from the SLM plane to the sensor plane.
	#				Then, u_{slm,in} is found by applying Bandlimit(...) to u'_{slm,in}.
	#					i) Note that u'_{slm,in} is a dummy variable.
	#					ii) Note that the MSE might not go to zero if u_{sensor} (equivalently u_{im}) is modeled as having
	#						high-frequency components.  If bandlimiting occcurs, then these high-frequency components might not be
	#						able to propagate from the SLM plane to the sensor plane.  Thus, in that case, there would be some discrepancy
	#						which would result in the MSE not going to zero.
	#			b) Prop(...) describes propagation from the SLM plane to the sensor plane
	#					i) Note that Prop(...) assumes that the SLM is absent, i.e. no phase modulation occurs
	#			# Note: The behavior of Bandlimit(...) depends on what slmInputBandlimitingFilter is.
	#			If slmInputBandlimitingFilter != None:
	#				c) Bandlimit(...) performs bandlimiting
	#					i) Bandlimiting is done by convolving u_{im} with slmInputBandlimitingFilter
	#			Else:
	#				c) Bandlimit(...) does nothing and can be ignored in the equations
	#	Else:
	#		2. The code does not handle this case
	#	4. For every i \in \{1, 2, \cdots, N\}, compute: u_{sensor_i} = Prop(u_{slm_i,out}) = Prop(SLM_i(u_{slm,in}))
	#		a) This calculates the fields u_{sensor_i} at the sensor plane assuming that the field u_{slm,in} is at the input of the SLM plane
	#			i) u_{slm,in} is the same for all SLM patterns as it represents the field that would be present at the SLM plane with no SLM present
	#		b) u_{slm_i,out} is the field at the output of the SLM plane for the i-th SLM pattern
	#		c) SLM_i(...) applies random pointwise phase shifts corresponding to the i-th SLM pattern
	#		d) Prop(...) describes propagation from the output of the SLM plane to the sensor plane
	#			i) This should be the same as the Prop(...) mentioned in (3b)
	#	5. For every i \in \{1, 2, \cdots, N\}, compute: (I_{sensor_i})^(1/2) = |u_{sensor_i}|
	#		a) This gives the square roots of the resulting intensity patterns at the sensor plane for each SLM pattern
	#		b) These are speckle patterns.
	#	FINAL OUTPUT:
	#		- N speckle patterns on the sensor plane for each of the N SLM patterns
	############################################################################
################################################################################################################################


class Synthetic_Data_Generator:

	def __init__(self,
					save_data_directory				: Path or str,
					creator_script_path				: str,

					sensorPlaneResolution			: tuple or list,
					sensorPlanePixelPitch			: SpacingContainer or float,
					slmPlaneResolution				: tuple or list,
					slmPlanePixelPitch				: SpacingContainer or float,
					wavelengths 					: WavelengthContainer or float,
					slmToSensorPlanePropModel		: CGH_Component,
					slmModel						: CGH_Component,
					detectorModel					: CGH_Component,
					inputImageFilepath				: str,
					objectDepthRange				: float = None,

					############################################################################
					# This determines whether the image is formed on the
					# sensor plane or the SLM plane.
					############################################################################
					#	imagePlaneType = 'sensor' 	<-- Assumes light rays are focused on the sensor plane.
					#									Fields at SLM plane are recovered using optimization.
					#
					#	imagePlaneType = 'slm'		<-- Assumes light are rays focused on the SLM plane.
					#									No need to recover the fields at the SLM plane.
					############################################################################
					imagePlaneType					: string = 'sensor',
					
					############################################################################
					# Settings for image plane field simulation/calculation
					############################################################################
					# NOTES:
					#	1. The points on the object that are closest to the sensor/lens lie
					#		in the plane at z = -(objectPlaneLocation+objectDistanceDelta),
					#		where negative z-coordinates are in front of the lens and positive
					#		z-coordinates are behind the lens.
					############################################################################
					imagePlaneCalculationType		: str = None,
					focusingLensFocalLength			: float = None,
					objectPlaneDistance				: float = None,
					imagePlaneDistance				: float = None,
					objectDistanceDelta				: float = 0,
					############################################################################

					############################################################################
					# Settings for image plane field simulations involving propagation
					############################################################################
					# NOTES:
					#	- These settings only have an effect if 'imagePlaneCalculationType' is one of
					#		these values: 'propagation_1'
					#	- 'objectPlaneResolution':
					#		- Should be a 2-tuple or list of length 2
					# 		- The fields at the object plane will have spatial dimensions (objectPlaneResolution[0]*objectPlaneSampleSpacing, objectPlaneResolution[1]*objectPlaneSampleSpacing).
					#	- 'objectPlaneSampleSpacing':
					#		- Determines the spacing of sample points in the
					#			object plane.  The same spacing will be used for x and y.
					#	- 'desiredObjectDimensions':
					#		- The input image will (roughly) specify fields in a region of space of dimension 'desiredObjectDimensions'
					#			- The aspect ratio of the image will not be distorted.  Images will be resized to best fill the region without exceeding it.
					#		- Example:
					#			- If desiredObjectDimensions=(15*mm,20*mm), objectPlaneSampleSpacing is 5*um, and the input image is 200x200 pixels, then
					#				the input image will be resized to 3000x3000 pixels, which corresponds to a 15mm x 15mm rectangle.
					#			- The resized image will be used to specify the fields in a 15mm x 15mm region in the object plane fields.
					############################################################################
					objectPlaneResolution			: tuple or list = None,
					objectPlaneSampleSpacing		: float = None,
					desiredObjectDimensions			: tuple or list = None,
					objectPlaneToImagePlaneModel	: CGH_Component = None,
					############################################################################

					slmInputBandlimitingFilter		: torch.tensor = None,
					saveSensorMeasurementsInMemory	: bool = False,
					useFloat16ForIntensity			: bool = True,

					extraMetadataDictionary			: dict = None,
					
					device							: torch.device = None,
					gpu_no							: int = 0,
					use_cuda						: bool = False,

					_debug_skip_long_ops			: bool = False
				) -> None:

		if (device != None):
			self.device = device
		else:
			self.device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
			self.gpu_no = gpu_no
			self.use_cuda = use_cuda

		self.creator_script_path = creator_script_path

		imagePlaneCalculationTypesList = ['basic_1', 'ray_optics_1', 'propagation_1']
		self._imagePlaneCalculationTypesList = imagePlaneCalculationTypesList
		if (imagePlaneCalculationType in imagePlaneCalculationTypesList):
			self.imagePlaneCalculationType = imagePlaneCalculationType
		else:
			raise Exception("Invalid value for 'imagePlaneCalculationType' given.  Should be 'basic_1', 'ray_optics_1', or 'propagation_1'.")

		if (imagePlaneCalculationType in ['ray_optics_1', 'propagation_1']):
			if (imagePlaneDistance is None) or (objectPlaneDistance is None) or (focusingLensFocalLength is None):
				raise Exception("Must specify imagePlaneDistance, objectPlaneLocation, and focusingLensFocalLength when imagePlaneCalculationType is set to " + imagePlaneCalculationType + ".")


		if len(sensorPlaneResolution) != 2:
			raise ValueError("Sensor plane dimensions must be 2D")
		self.sensorPlaneResolution = sensorPlaneResolution

		if len(slmPlaneResolution) != 2:
			raise ValueError("SLM plane dimensions must be 2D")	
		self.slmPlaneResolution = slmPlaneResolution

		if (desiredObjectDimensions is not None):
			if (len(desiredObjectDimensions) != 2):
				raise ValueError("'desiredObjectDimensions' should be a 2-element tuple or list.")
		elif (imagePlaneCalculationType == 'propagation_1'):
			raise ValueError("'desiredObjectDimensions' must be defined if imagePlaneCalculationType is 'propagation_1'.")
		self.desiredObjectDimensions = desiredObjectDimensions

		if torch.is_tensor(wavelengths):
			raise Exception("Not implemented.")
		elif np.isscalar(wavelengths):
			wavelengths = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=1))
		elif isinstance(wavelengths, list):
			wavelengths = WavelengthContainer(wavelengths=wavelengths, tensor_dimension=Dimensions.C(n_channel=np.size(wavelengths)))
		else:
			raise Exception("Not implemented.")

		# Convert wavelengths to 6D tensor
		self.wavelengths_BTPCHW  = wavelengths.data_tensor.view(wavelengths.tensor_dimension.get_new_shape(new_dim=Dimensions.BTPCHW))

		# Move wavelengths_BTPCHW to device
		self.wavelengths_BTPCHW = self.wavelengths_BTPCHW.to(device=self.device)
		
		if (isinstance(sensorPlanePixelPitch, SpacingContainer)):
			raise Exception("Not implemented.")
		elif np.isscalar(sensorPlanePixelPitch):
			sensorPlanePixelPitch = SpacingContainer(spacing=sensorPlanePixelPitch) # Should default to TCD dimensions
		else:
			raise Exception("'sensorPlanePixelPitch' must be a scalar.")

		if (isinstance(slmPlanePixelPitch, SpacingContainer)):
			raise Exception("Not implemented.")
		elif np.isscalar(slmPlanePixelPitch):
			slmPlanePixelPitch = SpacingContainer(spacing=slmPlanePixelPitch) # Should default to TCD dimensions
		else:
			raise Exception("'slmPlanePixelPitch' must be a scalar.")

		self.wavelengths = wavelengths.to(device=self.device)
		
		self.sensorPlanePixelPitch = sensorPlanePixelPitch.to(device=self.device)
		self.slmPlanePixelPitch = slmPlanePixelPitch.to(device=self.device)

		if (slmInputBandlimitingFilter is None):
			self.slmInputBandlimitingFilter = None
		else:
			self.slmInputBandlimitingFilter = slmInputBandlimitingFilter.to(device=self.device)


		self.inputImageFilepath = inputImageFilepath
		if not Path(inputImageFilepath).exists():
			raise Exception("The input image filepath given (" + str(inputImageFilepath) + ") does not point to a valid location.")

		
		# impliedDepthRange = self.checkForImpliedDepthRangeInFilename(inputImageFilepath)
		# if (impliedDepthRange is not None):
		# 	if (objectDepthRange is None):
		# 		self.objectDepthRange = impliedDepthRange
		# 	else:
		# 		print("The input image filename (%s) implies an object depth range of %g m, but the 'objectDepthRange' argument provided to this class specifies a depth range of %g m." % (Path(inputImageFilepath).parts[-1], impliedDepthRange, objectDepthRange))
		# 		print("Which value for object depth range would you like to use?")
		# 		print("\t0: %g from the the input image filename" % (impliedDepthRange))
		# 		print("\t1: %g from the the 'objectDepthRange' argument" % (objectDepthRange))
		# 		while True:
		# 			resp = input("Please select an option (enter 0 or 1): ")
		# 			if (resp == '0'):
		# 				self.objectDepthRange = impliedDepthRange
		# 				print("Using %g m for the object depth range." % (impliedDepthRange))
		# 				break
		# 			elif (resp == '1'):
		# 				self.objectDepthRange = objectDepthRange
		# 				print("Using %g m for the object depth range." % (objectDepthRange))
		# 				break
		# 			else:
		# 				print("Invalid input.  Input should be either 0 or 1.")
		# elif (objectDepthRange is not None):
		# 	self.objectDepthRange = objectDepthRange
		# else:
		# 	raise Exception("No depth range was specified by the user, and no depth range was implied by the input image filename.  Cannot continue.")

		# # Overwriting the objectDepthRange variable to prevent accidental use of the wrong value:
		# objectDepthRange = self.objectDepthRange

		if (objectDepthRange is not None):
			self.objectDepthRange = objectDepthRange
		else:
			raise Exception("No depth range was specified.  Please specify a value for 'objectDepthRange'.")


		if (imagePlaneType == 'sensor'):
			self.imagePlaneResolution = sensorPlaneResolution
			self.imagePlanePixelPitch = sensorPlanePixelPitch.to(device=self.device)
		elif (imagePlaneType == 'slm'):
			self.imagePlaneResolution = slmPlaneResolution
			self.imagePlanePixelPitch = slmPlanePixelPitch.to(device=self.device)
		else:
			raise Exception("Invalid type for 'imagePlaneType'.")

		self.imagePlaneType = imagePlaneType

		self.focusingLensFocalLength = focusingLensFocalLength
		self.objectPlaneDistance = objectPlaneDistance
		self.imagePlaneDistance = imagePlaneDistance
		self.objectDistanceDelta = objectDistanceDelta
		
		self.objectPlaneResolution = objectPlaneResolution
		self.objectPlaneSampleSpacing = objectPlaneSampleSpacing

		if (focusingLensFocalLength <= 0):
			raise Exception("'focusingLensFocalLength' should be > 0 a a converging lense is assumed.")
		if (imagePlaneDistance <= 0):
			raise Exception("The image plane should probably be behind the lens (i.e. imagePlaneDistance > 0).")
		if ((objectPlaneDistance + objectDistanceDelta) <=0):
			raise Exception("The object plane should be located in front of the lens (i.e. objectPlaneLocation+objectPlaneDelta > 0).")
		elif ((objectPlaneDistance + objectDistanceDelta) <= focusingLensFocalLength):
			raise Exception("One should probably have a real image (i.e. objectPlaneLocation+objectPlaneDelta > focusingLensFocalLength).")


		tempThreshold = 0.01
		calculatedimagePlaneDistance1 = self.calc_image_plane_loc(objectPlaneDistance + objectDistanceDelta, focusingLensFocalLength)
		calculatedimagePlaneDistance2 = self.calc_image_plane_loc(objectPlaneDistance + objectDistanceDelta + self.objectDepthRange, focusingLensFocalLength)
		if ((abs(calculatedimagePlaneDistance1 - self.imagePlaneDistance) / self.imagePlaneDistance) > tempThreshold) or ((abs(calculatedimagePlaneDistance2 - self.imagePlaneDistance) / self.imagePlaneDistance) > tempThreshold):
			warnings.warn("The range of object depths is great enough that the image plane location might vary too much over that range.")


		self.saveSensorMeasurementsInMemory = saveSensorMeasurementsInMemory
		self.useFloat16ForIntensity = useFloat16ForIntensity

		self.slmToSensorPlanePropModel = slmToSensorPlanePropModel
		self.slmModel = slmModel
		self.detectorModel = detectorModel
		self.objectPlaneToImagePlaneModel = objectPlaneToImagePlaneModel

		# Creates model that describes going from the SLM input to the detected intensity signal at the sensor
		self.slmInputToSensorOutputModel = torch.nn.Sequential(self.slmModel, self.slmToSensorPlanePropModel, self.detectorModel)

		if (save_data_directory is None):
			self.save_data_directory = None
			warnings.warn("No output folder for the data is specified.  Data will not be saved.")
		elif (isinstance(save_data_directory, str)):
			self.save_data_directory = Path(save_data_directory)
		elif (isinstance(save_data_directory, Path)):
			self.save_data_directory = save_data_directory
		else:
			raise Exception("Error: Invalid input for 'data_output_dir'.")

		if (extraMetadataDictionary is None):	# Could probably just set the default value of extraMetadataDictionary to be an empty dictionary instead
			self.extraMetadataDictionary = {}
		else:
			self.extraMetadataDictionary = extraMetadataDictionary

		self._debug_skip_long_ops = _debug_skip_long_ops

		self.calc_xy_grids_image_plane()
		self.calc_xy_grids_object_plane()


		if (self.imagePlaneCalculationType == 'propagation_1'):
			targetImageResolution = (int(desiredObjectDimensions[0] // objectPlaneSampleSpacing), int(desiredObjectDimensions[1] // objectPlaneSampleSpacing))
			if (targetImageResolution[0] > objectPlaneResolution[0]) or (targetImageResolution[1] > objectPlaneResolution[1]):
				raise Exception("'desiredObjectDimensions' specifies a space that is too big to fit in the object plane.")
			self.load_image_file(targetResolution=targetImageResolution)
		else:
			self.load_image_file(targetResolution=self.imagePlaneResolution)

		if (self.imagePlaneCalculationType in ['ray_optics_1', 'propagation_1']):
			self.delta_x_depth = self.delta_x_object_plane
			self.delta_y_depth = self.delta_y_object_plane
			self.xGridDepth = self.xGridObjectPlane
			self.yGridDepth = self.yGridObjectPlane
		elif (self.imagePlaneCalculationType == 'basic_1'):
			self.delta_x_depth = self.delta_x_image_plane
			self.delta_y_depth = self.delta_y_image_plane
			self.xGridDepth = self.xGridImagePlane
			self.yGridDepth = self.yGridImagePlane

		self.get_depth_and_reflectance_from_image()
		self.calculate_path_lengths()

		if (self.imagePlaneCalculationType == 'propagation_1'):
			if (self.objectPlaneToImagePlaneModel is None):
				raise Exception("Must specify a model for 'objectPlaneToImagePlaneModel' if 'imagePlaneCalculationType' is 'propagation_1'.")
			self.compute_field_on_object_plane()
			self.createPropagation1Model()
		else:
			self.objectPlaneField = None
			self.objectPlaneToImagePlaneModel = None

		self.compute_field_on_image_plane()

		self.objectMeshGroundTruth = self.get_3D_object_mesh(4)

		if (self._debug_skip_long_ops):
			print("NOTE: _debug_skip_long_ops is set to True.  Skipping SLM input field and sensor plane measurement calculations.  Not saving any data to disk.")
			self.slmInputPlaneField = None
			self.sensorPlaneFieldEstimateFromSlmPlane = None
		else:
			self.calculate_slm_input_fields()
			self.initialize_save_folder()
			self.calculate_and_save_synthetic_data()


	def calc_xy_grids_image_plane(self):
		# Old:
		# xCoords = torch.linspace(-((self.imagePlaneResolution[0] - 1) // 2), (self.imagePlaneResolution[0] - 1) // 2, self.imagePlaneResolution[0])
		# yCoords = torch.linspace(-((self.imagePlaneResolution[1] - 1) // 2), (self.imagePlaneResolution[1] - 1) // 2, self.imagePlaneResolution[1])
		# xGrid, yGrid = torch.meshgrid(xCoords, yCoords)
		# xGrid = xGrid.to(device=self.device)
		# yGrid = yGrid.to(device=self.device)
		# xGrid = xGrid * delta_x
		# yGrid = yGrid * delta_y
		# self.xCoordsImagePlane = xCoords
		# self.yCoordsImagePlane = yCoords

		try:
			heightDimInd = self.imagePlanePixelPitch.tensor_dimension.id.index(Dimensions.DIM.HEIGHT)
		except:
			raise Exception("Height dimension not present in spacing data tensor.")
		
		if (self.imagePlanePixelPitch.shape[heightDimInd] != 2):
			raise Exception("Spacing should be defined for two dimensions.")
		
		spacingDataTemp = self.imagePlanePixelPitch.data_tensor.view(self.imagePlanePixelPitch.tensor_dimension.get_new_shape(new_dim=Dimensions.BTPCHW))
		delta_x = spacingDataTemp[:,:,:,:,0,:]
		delta_y = spacingDataTemp[:,:,:,:,1,:]

		xGrid, yGrid = generateGrid(self.imagePlaneResolution, delta_x, delta_y, device=self.device)
		
		self.delta_x_image_plane = delta_x
		self.delta_y_image_plane = delta_y
		self.xGridImagePlane = xGrid
		self.yGridImagePlane = yGrid


	def calc_xy_grids_object_plane(self):
		if self.imagePlaneCalculationType == 'basic_1':
			self.delta_x_object_plane = None
			self.delta_y_object_plane = None
			self.xGridObjectPlane = None
			self.yGridObjectPlane = None
		elif self.imagePlaneCalculationType == 'ray_optics_1':
			magnification = self.calc_magnification(self.objectPlaneDistance)

			delta_x_object_plane = self.delta_x_image_plane * magnification
			delta_y_object_plane = self.delta_y_image_plane * magnification
			xGridObjectPlane = self.xGridImagePlane * magnification
			yGridObjectPlane = self.yGridImagePlane * magnification

			self.delta_x_object_plane = delta_x_object_plane
			self.delta_y_object_plane = delta_y_object_plane
			self.xGridObjectPlane = xGridObjectPlane
			self.yGridObjectPlane = yGridObjectPlane
		elif self.imagePlaneCalculationType == 'propagation_1':
			xGridObjectPlane, yGridObjectPlane = generateGrid(self.objectPlaneResolution, self.objectPlaneSampleSpacing, self.objectPlaneSampleSpacing, device=self.device)
			self.delta_x_object_plane = self.objectPlaneSampleSpacing
			self.delta_y_object_plane = self.objectPlaneSampleSpacing
			self.xGridObjectPlane = xGridObjectPlane
			self.yGridObjectPlane = yGridObjectPlane
		else:
			raise Exception("Should not be in this state.")

	
	def calculate_path_lengths(self):
		if (self.imagePlaneCalculationType == 'ray_optics_1'):
			dx = self.xGridImagePlane - self.xGridObjectPlane
			dy = self.yGridImagePlane - self.yGridObjectPlane
			dz = self.imagePlaneDistance + self.objectPlaneDistance + self.depths
			R = torch.sqrt((dx**2) + (dy**2) + (dz**2))
			
			self.pathLengths = R
		elif (self.imagePlaneCalculationType == 'basic_1'):
			if (self.objectPlaneDistance is not None) and (self.imagePlaneDistance is not None):
				self.pathLengths = self.depths + self.objectPlaneDistance + self.imagePlaneDistance
			else:
				self.pathLengths = self.depths
		elif (self.imagePlaneCalculationType == 'propagation_1'):
			self.pathLengths = None


	def calc_image_plane_loc(self, z_obj, focalLen = None):
		if (focalLen is not None):
			f = focalLen
		elif (self.focusingLensFocalLength is not None):
			f = self.focusingLensFocalLength
		else:
			raise Exception("No focal length defined.  Cannot calculate image plane locations.")
		z_image = -z_obj * f / (f - z_obj)
		return z_image


	def calc_magnification(self, z_obj):
		if (self.focusingLensFocalLength is None):
			raise Exception("No focal length defined.  Cannot calculate magnifications.")
		f = self.focusingLensFocalLength
		mag = f / (f - z_obj)
		return mag


	# Note: This method has been deprecated and replaced with get_field_slice(...) from holotorch_addons.HelperFunctions.
	# Note: This method is not robust and does not handle every possible case
	def get_field_slice_channel(self, field, channelNum):
		# Assumes that the field has dimensions BTPCHW and that the B, T, and P dimensions have size 1
		# Assumes that the wavelength container has dimensions C
		# Assumes that the spacing container has dimensions TCD
		if not (np.size(field.data.size()) == 6):
			raise Exception("Field tensor dimensions should be BTPCHW.")
		if not (isinstance(field.wavelengths.tensor_dimension, Dimensions.C)):
			raise Exception("Wavelength container dimensions should be C.")
		if not (isinstance(field.spacing.tensor_dimension, Dimensions.TCD)):
			raise Exception("Spacing container dimensions should be TCD.")
		
		if not (np.array_equal(self.imagePlaneField.shape[0:3], [1, 1, 1])):
			raise Exception("Unexpected input.  The B, T, and/or P dimension(s) of field data tensor are not singleton dimensions.")

		new_field_data = field.data[:,:,:,channelNum,:,:]
		new_field_data = new_field_data[:,:,:,None,:,:]		# Expand back to 6D tensor

		new_wavelength_container =	WavelengthContainer(
										wavelengths = field.wavelengths.data_tensor[channelNum],
										tensor_dimension = Dimensions.C(n_channel=1)
									)

		new_spacing_container = field.spacing	# Assumes that the spacing container's dimensions are TCD(n_time=1,n_channel=1,height=2), e.g. same spacing for all channels

		newField = 	ElectricField(
						data = new_field_data,
						wavelengths = new_wavelength_container.to(device=self.device),
						spacing = new_spacing_container.to(device=self.device)
					)

		return newField


	def plot_depth_map(self):
		pass


	def plot_plane_fields_single_channel(self, plot_channel, rescale_factor = 0.25):
		raise Exception("Not implemented")
		
		def plot_helper(field, magnitudeTitle, phaseTitle):
			if (numCols <= 0):
				raise Exception("Error in the code: did not properly set numCols.")
			nonlocal curCol
			plt.subplot(2, numCols, curCol)
			get_field_slice(field, channel_inds_range=plot_channel, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor = rescale_factor, flag_colorbar=True, flag_axis=True, plot_type=ENUM_PLOT_TYPE.MAGNITUDE)
			plt.title(magnitudeTitle)
			plt.subplot(2, numCols, curCol + numCols)
			get_field_slice(field, channel_inds_range=plot_channel, field_data_tensor_dimension=Dimensions.BTPCHW).visualize(rescale_factor = rescale_factor, flag_colorbar=True, flag_axis=True, plot_type=ENUM_PLOT_TYPE.PHASE)
			plt.title(phaseTitle)
			curCol += 1

		# Variables for determining which plots to show
		show_ground_truth_u_slm			= plot_helper(	self.slmInputPlaneField,
														'"Ground truth" SLM plane field (Magnitude)\n' + r'$\left\|u_{slm}\right\|$',
														'"Ground truth" SLM plane field (Phase)\n' + r'$\mathrm{Arg}\left\{u_{slm}\right\}$'	)
		show_estimated_u_slm_1			= plot_helper(	self.slmInputPlaneField,
														'Estimated SLM plane field (Magnitude)\n' + r'$\left\|\hat{u}_{slm}\right\| = \left\|\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}^{^{-1}}\left(u_{sensor}\right)\right\|$',
														'Estimated SLM plane field (Phase)\n' + r'$\mathrm{Arg}\left\{\hat{u}_{slm}\right\} = \mathrm{Arg}\left\{\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}^{^{-1}}\left(u_{sensor}\right)\right\}$'	)
		show_ground_truth_u_sensor		= plot_helper(	self.imagePlaneField,
														'"Ground truth" sensor plane field (Magnitude)\n' + r'$\left\|u_{sensor}\right\|$',
														'"Ground truth" sensor plane field (Phase)\n' + r'$\mathrm{Arg}\left\{u_{sensor}\right\}$'	)
		show_estimated_u_sensor_1		= plot_helper(	self.sensorPlaneFieldEstimateFromSlmPlane,
														'Estimated sensor plane field (Magnitude)\n' + r'$\left\|\hat{u}_{sensor}\right\| = \left\|\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}\left(\hat{u}_{slm}\right)\right\|$',
														'Estimated sensor plane field (Phase)\n' + r'$\mathrm{Arg}\left\{\hat{u}_{sensor}\right\} = \mathrm{Arg}\left\{\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}\left(\hat{u}_{slm}\right)\right\}$'	)

		# Initializing variables
		numCols = -1
		curCol = 1

		if (self.imagePlaneCalculationType == 'propagation_1'):
			numCols = 3
			plot_helper(	self.objectPlaneField,
							'"Ground truth" object plane field (Magnitude)\n' + r'$\left\|u_{obj}\right\|$',
							'"Ground truth" object plane field (Phase)\n' + r'$\mathrm{Arg}\left\{u_{obj}\right\}$'	)
		
			if (self.imagePlaneType == 'sensor'):
				plot_helper(	self.imagePlaneField,
								'Sensor plane field (Magnitude)\n' + r'$\left\|u_{sensor}\right\| = \left\|\mathrm{PROP}_{u_{obj} \rightarrow u_{sensor}}\left(u_{obj}\right)\right\|$',
								'Sensor plane field (Phase)\n' + r'$\mathrm{Arg}\left\{u_{sensor}\right\} = \mathrm{Arg}\left\{\mathrm{PROP}_{u_{obj} \rightarrow u_{sensor}}\left(u_{obj}\right)\right\}$'	)
				plot_helper(	self.slmInputPlaneField,
								'Estimated SLM plane field (Magnitude)\n' + r'$\left\|\hat{u}_{slm}\right\| = \left\|\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}^{^{-1}}\left(\hat{u}_{sensor}\right)\right\|$',
								'Estimated SLM plane field (Phase)\n' + r'$\mathrm{Arg}\left\{\hat{u}_{slm}\right\} = \mathrm{Arg}\left\{\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}^{^{-1}}\left(\hat{u}_{sensor}\right)\right\}$'	)
				

			elif (self.imagePlaneType == 'slm') and (self.imagePlaneCalculationType == 'propagation_1'):
				plot_helper(	self.imagePlaneField,
								'SLM plane field (Magnitude)\n' + r'$\left\|u_{slm}\right\| = \left\|\mathrm{PROP}_{u_{obj} \rightarrow u_{slm}}\left(u_{obj}\right)\right\|$',
								'SLM plane field (Phase)\n' + r'$\mathrm{Arg}\left\{u_{slm}\right\} = \mathrm{Arg}\left\{\mathrm{PROP}_{u_{obj} \rightarrow u_{slm}}\left(u_{obj}\right)\right\}$'	)
				plot_helper(	self.imagePlaneField,
								'Sensor plane field (Magnitude)\n' + r'$\left\|u_{sensor}\right\| = \left\|\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}\left(u_{slm}\right)\right\|$',
								'Sensor plane field (Phase)\n' + r'$\mathrm{Arg}\left\{u_{sensor}\right\} = \mathrm{Arg}\left\{\mathrm{PROP}_{u_{slm} \rightarrow u_{slm}}\left(u_{slm}\right)\right\}$'	)
		elif (self.imagePlaneCalculationType in ['basic_1', 'ray_optics_1']) and (self.imagePlaneType == 'sensor'):
			numCols = 3
			plot_helper(	self.imagePlaneField,
							'"Ground truth" sensor plane field (Magnitude)\n' + r'$\left\|u_{sensor}\right\|$',
							'"Ground truth" sensor plane field (Phase)\n' + r'$\mathrm{Arg}\left\{u_{sensor}\right\}$'	)
			plot_helper(	self.slmInputPlaneField,
							'Estimated SLM plane field (Magnitude)\n' + r'$\left\|\hat{u}_{slm}\right\| = \left\|\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}^{^{-1}}\left(\hat{u}_{sensor}\right)\right\|$',
							'Estimated SLM plane field (Phase)\n' + r'$\mathrm{Arg}\left\{\hat{u}_{slm}\right\} = \mathrm{Arg}\left\{\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}^{^{-1}}\left(\hat{u}_{sensor}\right)\right\}$'	)
			plot_helper(	self.imagePlaneField,
							'Sensor plane field (Magnitude)\n' + r'$\left\|\hat{u}_{sensor}\right\| = \left\|\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}\left(\hat{u}_{slm}\right)\right\|$',
							'Sensor plane field (Phase)\n' + r'$\mathrm{Arg}\left\{\hat{u}_{sensor}\right\} = \mathrm{Arg}\left\{\mathrm{PROP}_{u_{slm} \rightarrow u_{sensor}}\left(\hat{u}_{slm}\right)\right\}$'	)
		
		if (self.imagePlaneCalculationType == 'propagation_1'):
			pass
		elif (self.imagePlaneCalculationType in ['basic_1', 'ray_optics_1']):
			pass
		else:
			raise Exception("Should not be in this state.")


	# Unused/removed, but saved just in case it's needed later
	# def checkForImpliedDepthRangeInFilename(self, filepathStr):
	# 	def getSuffixStr(str):
	# 		underscoreIndex = None
	# 		for match in re.finditer('_', str):
	# 			underscoreIndex = match.start()
			
	# 		if underscoreIndex is None:
	# 			return None
			
	# 		suffixStrTemp = str[underscoreIndex+1:]
	# 		if len(suffixStrTemp) == 0:
	# 			return None

	# 		dotIndex = None
	# 		for match in re.finditer('\.', suffixStrTemp):
	# 			dotIndex = match.start()

	# 		if dotIndex is None:
	# 			return suffixStrTemp
	# 		suffixStrTemp2 = suffixStrTemp[0:dotIndex]
	# 		if len(suffixStrTemp2) == 0:
	# 			return None
	# 		return suffixStrTemp2

	# 	tempPath = Path(filepathStr)
	# 	if (len(tempPath.parts) == 0):
	# 		return None
	# 	filenameStr = tempPath.parts[-1]
	# 	suffixStr = getSuffixStr(filenameStr)
	# 	if (suffixStr is None):
	# 		return None
	# 	try:
	# 		value, _, unitType = parseNumberAndUnitsString(suffixStr)
	# 		if (unitType != 'spatial'):
	# 			return None
	# 		else:
	# 			return value
	# 	except:
	# 		return None


	def load_image_file(self, targetResolution):
		inputImage = Image.fromarray(imageio.imread(self.inputImageFilepath))
		self.inputImageObject = fit_image_to_resolution(inputImage, targetResolution)

		if (list(inputImage.size) != list(targetResolution)):
			print("NOTE: that the input image was resized to best fit the resolution (Height,Width) = (%d,%d) while still preserving the input image's aspect ratio.  If the aspect ratios did not match, then zero-padding was applied." % (targetResolution[0], targetResolution[1]))


	def get_depth_and_reflectance_from_image(self):
		inputImage = torch.tensor(np.array(self.inputImageObject), device=self.device)

		# The red channel of the input image describes the depths at various points on the imaged object.
		# Red (depth) channel values of 255 are mapped to depths of zero,
		# and values of 0 are mapped to depths of 'self.inputImageDepthRange'.
		depthChannelNormalized = inputImage[:,:,0] / 255	# inputImage[:,:,0] corresponds to the red color channel
		depths = depthChannelNormalized - torch.min(depthChannelNormalized.abs())
		if (torch.max(depths.abs()) != 0):
			depths = depths / torch.max(depths.abs())
		else:
			depths[:,:] = 1
		depths = (self.objectDepthRange * (1 - depths)) + self.objectDistanceDelta

		# The green channel of the input image describes the reflectance/albedo at various points on the imaged object.
		# Green (reflectance/albedo) channel values of 255 correspond to 100% reflection,
		# and values of 0 correspond to 0% reflection.
		# Note that the reflectance is limited to a positive scalar on the interval [0,1]
		reflectanceChannel = inputImage[:,:,1] / 255		# inputImage[:,:,1] corresponds to the green color channel

		if (self.imagePlaneCalculationType != 'propagation_1'):
			self.depths = depths.to(self.device)
			self.reflectances = reflectanceChannel.to(self.device)
		else:
			# Make the depths and reflectances tensors fit the resolution of the object plane
			objPlaneRes = self.objectPlaneResolution
			imgRes = list(depths.shape)

			pad_x = (objPlaneRes[0] - imgRes[0]) / 2
			pad_x0 = int(np.floor(pad_x))
			pad_x1 = int(np.ceil(pad_x))

			pad_y = (objPlaneRes[1] - imgRes[1]) / 2
			pad_y0 = int(np.floor(pad_y))
			pad_y1 = int(np.ceil(pad_y))

			depths = torch.nn.functional.pad(depths, (pad_y0, pad_y1, pad_x0, pad_x1), mode='constant', value=0)
			reflectanceChannel = torch.nn.functional.pad(reflectanceChannel, (pad_y0, pad_y1, pad_x0, pad_x1), mode='constant', value=0)

			self.depths = depths.to(self.device)
			self.reflectances = reflectanceChannel.to(self.device)


	def get_3D_object_mesh(self, subsamplingMagnitude, invertDepths = True, gridXSignMultiplier = -1, gridYSignMultiplier=-1, setMinDepthReferenceDepth = True, minDepthReferenceDepth = 0):
		if (self.imagePlaneCalculationType == 'basic_1'):
			gridX = self.xGridImagePlane
			gridY = self.yGridImagePlane
		elif (self.imagePlaneCalculationType in ['ray_optics_1', 'propagation_1']):
			gridX = self.xGridObjectPlane
			gridY = self.yGridObjectPlane
		else:
			raise Exception("Not implemented")

		gridX = gridX * np.sign(gridXSignMultiplier)
		gridY = gridY * np.sign(gridYSignMultiplier)

		depths = self.depths.clone()
		if invertDepths:
			depths = -depths
		if setMinDepthReferenceDepth:
			depths = depths - depths.min() + minDepthReferenceDepth

		# groundTruthDepthData = {'depths': depths, 'gridX': gridX, 'gridY': gridY}
		# torch.save(groundTruthDepthData, 'groundTruthDepthData.pt')
		
		return createMeshFromGridsAndDepths(gridX, gridY, depths, subsamplingMagnitude=subsamplingMagnitude)


	def createPropagation1Model(self):
		outputResampler =	Field_Resampler(
								outputHeight = int(self.imagePlaneResolution[0]),
								outputWidth = int(self.imagePlaneResolution[1]),

								# If the imagePlanePixelPitch spacing container does not have dimensions TxCxD with sizes 1x1x2, the
								# next two lines will probably eventually cause an error.
								outputPixel_dx = float(self.imagePlanePixelPitch.data_tensor[:,:,0]),
								outputPixel_dy = float(self.imagePlanePixelPitch.data_tensor[:,:,1]),
								
								device = self.device
							)
		self.objectPlaneToImagePlaneModel_WithResampling = torch.nn.Sequential(self.objectPlaneToImagePlaneModel, outputResampler)


	def compute_field_on_object_plane(self):
		field_data_tensor = self.reflectances * torch.exp(1j*2*np.pi*self.depths/self.wavelengths_BTPCHW)
		self.objectPlaneField = ElectricField(
									data = field_data_tensor,
									wavelengths = self.wavelengths,
									spacing = self.objectPlaneSampleSpacing
								)
		self.objectPlaneField.spacing = self.objectPlaneField.spacing.to(self.device)


	def compute_propagation1_image_plane_field(self):
		# self.imagePlaneField = self.objectPlaneToImagePlaneModel_WithResampling(self.objectPlaneField)
		fieldDataTensor = torch.zeros([1,1,1,self.wavelengths.channel,int(self.imagePlaneResolution[0]),int(self.imagePlaneResolution[1])], dtype=torch.cfloat, device='cpu')
		for i in range(self.wavelengths.channel):
			channelField = 	self.objectPlaneToImagePlaneModel_WithResampling(
								get_field_slice(
													self.objectPlaneField,
													channel_inds_range=i,
													field_data_tensor_dimension=Dimensions.BTPCHW,
													cloneTensors=False,
												)
							)
			channelField.data = channelField.data.to(device='cpu')
			fieldDataTensor[:,:,:,i,:,:] = channelField.data
			del channelField

		self.imagePlaneField = 	ElectricField(
												data = fieldDataTensor.to(device=self.device),
												wavelengths = self.wavelengths,
												spacing = self.imagePlanePixelPitch
											)
		torch.cuda.empty_cache()


	def compute_field_on_image_plane(self):
		if self.imagePlaneCalculationType in ['basic_1', 'ray_optics_1']:
			dimList = list(self.wavelengths_BTPCHW.size())
			dimList.extend([self.imagePlaneResolution[0], self.imagePlaneResolution[1]])
			# field_data_tensor = torch.zeros(dimList, dtype=torch.cfloat, device=self.device)
			field_data_tensor = self.reflectances * torch.exp(1j*2*np.pi*self.pathLengths/self.wavelengths_BTPCHW)
			self.imagePlaneField = 	ElectricField(
													data = field_data_tensor,
													wavelengths = self.wavelengths,
													spacing = self.imagePlanePixelPitch
												)
		elif self.imagePlaneCalculationType == 'propagation_1':
			self.compute_propagation1_image_plane_field()
		else:
			raise Exception("Should not be in this state.")

	
	def calculate_slm_input_fields(self):
		if (self.imagePlaneType == 'slm'):
			slmInputPlaneField = ElectricField(
								data = torch.tensor(self.imagePlaneField.data, dtype=torch.cfloat, device=self.device),
								wavelengths = self.wavelengths,
								spacing = self.slmPlanePixelPitch
							)

			if (self.slmInputBandlimitingFilter is None):
				self.slmInputPlaneField = slmInputPlaneField
			else:
				self.slmInputPlaneField = applyFilterToElectricField(self.slmInputBandlimitingFilter, slmInputPlaneField)
				
		elif (self.imagePlaneType == 'sensor'):
			# Optimization settings
			maxIterations = 500
			# maxIterations = 25 # For debugging
			plateauDeltaThreshold = 1e-7
			plateauIterationsThreshold = 20

			# Generate dimensions for 6D
			#	Assumes that B, T, and P dimensions are of size 1
			inputFieldSize = torch.Size([1, 1, 1, self.wavelengths.tensor_dimension.channel, self.slmPlaneResolution[0], self.slmPlaneResolution[1]])

			inputField = 	ElectricField(
								data = torch.zeros(inputFieldSize, dtype=torch.cfloat, device=self.device, requires_grad=True),
								wavelengths = self.wavelengths,
								spacing = self.slmPlanePixelPitch
							)

			optimizer = torch.optim.Adam([inputField.data], lr=0.3)
			numElems = self.imagePlaneField.data.numel()
			smallestLoss = np.Infinity
			prevLoss = np.Infinity
			numPlateauIterations = 0
			fieldsAtSlmPlane_BestFit = []

			print('Using optimization to figure out SLM fields...')
			print("")
			print('    Iteration\t|\tLoss (MSE)')
			print('--------------------------------------')

			for t in range(maxIterations):
				if (self.slmInputBandlimitingFilter is None):
					modelInput = inputField
				else:
					modelInput = applyFilterToElectricField(self.slmInputBandlimitingFilter, inputField)

				y = self.slmToSensorPlanePropModel(modelInput)
				L_fun = torch.sum(((y.data - self.imagePlaneField.data).abs()) ** 2) / numElems

				curLoss = L_fun.item()

				if (curLoss < smallestLoss):
					smallestLoss = curLoss
					# fieldsAtSlmPlane_BestFit = inputField.data
					fieldsAtSlmPlane_BestFit = modelInput.data

				# Zero gradients, perform a backward pass, and update the weights.
				optimizer.zero_grad()
				L_fun.backward()
				optimizer.step()

				print('\t%d\t|\t%.10f' % (t + 1, curLoss))

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
								data = torch.tensor(fieldsAtSlmPlane_BestFit, dtype=torch.cfloat, device=self.device), # Done this way so that requires_grad=False for this tensor
								wavelengths = self.wavelengths,
								spacing = self.slmPlanePixelPitch
							)
			
			self.slmInputPlaneField = slmInputPlaneField
			self.sensorPlaneFieldEstimateFromSlmPlane = self.slmToSensorPlanePropModel(slmInputPlaneField)
		else:
			raise Exception("Not implemented")


	def initialize_save_folder(self):
		if (self.save_data_directory is None):
			print("NOTE: Not saving data.")
			return
		
		os.makedirs(self.save_data_directory, exist_ok=True)

		curDateTime = datetime.datetime.today()
		saveFolderStr = 'DATA_' + str(curDateTime.year) + '-' + str(curDateTime.month) + '-' + str(curDateTime.day) + '_' + \
							str(curDateTime.hour).zfill(2) + 'h' + str(curDateTime.minute).zfill(2) + 'm' + str(curDateTime.second).zfill(2) + 's'
		self.saveDataFolder = self.save_data_directory / saveFolderStr
		self.saveDataSLMDataFolder = self.saveDataFolder / "SLM_Data"
		self.saveDataSensorDataFolder = self.saveDataFolder / "Sensor_Data"

		os.makedirs(self.saveDataFolder, exist_ok=False)
		print("Created save folder: %s" % (self.saveDataFolder.resolve()))
		os.makedirs(self.saveDataSLMDataFolder, exist_ok=False)
		print("Created folder for SLM data: %s" % (self.saveDataSLMDataFolder.resolve()))
		os.makedirs(self.saveDataSensorDataFolder, exist_ok=False)
		print("Created folder for sensor data: %s" % (self.saveDataSensorDataFolder.resolve()))

		self.dataFolderCreationTime = curDateTime


	def save_creator_script_to_directory(self):
		filepath = Path(self.creator_script_path)

		if not filepath.exists():
			raise Exception("Tried to read a file but the file does not exist.")

		f = open(filepath.resolve(), "r", encoding="utf-8")

		entireFileContents = ""
		saveContentsTemp = ""
		foundBeginMarker = False
		saveLine = False
		for line in f:
			entireFileContents += line
			if line.strip() == "#### BEGIN BACKUP REGION ####":
				saveLine = True
				foundBeginMarker = True
				continue
			if foundBeginMarker:
				if line.strip() == "#### END BACKUP REGION ####":
					break
			if saveLine:
				saveContentsTemp += line

		f.close()

		if foundBeginMarker:
			saveContents = saveContentsTemp
		else:
			saveContents = entireFileContents

		saveFilePath = (self.saveDataFolder / 'model_creation_script_backup.py.txt').resolve()
		fWrite = open(saveFilePath, "w", encoding="utf-8")
		fWrite.write(saveContents)
		fWrite.close()

		return saveFilePath


	def calculate_and_save_synthetic_data(self):
		n_batches = self.slmModel.n_slm_batches
		
		if (self.saveSensorMeasurementsInMemory):
			self.sensorMeasurements = [None] * n_batches

		# if not (self.device.type == 'cpu'):
		# 	clearCudaCacheFlag = True
		# else:
		# 	clearCudaCacheFlag = False
		
		# if (clearCudaCacheFlag):
		# 	torch.cuda.empty_cache() # Helps prevent the GPU memory from filling up
		# 	print_cuda_memory_usage(self.device)
		# 	print()

		if (self.save_data_directory is None):
			print("NOTE: Not saving SLM plane input field.")
		else:
			slmInputPlaneFieldPath = self.saveDataFolder / 'SLM_Input_Plane_Field.pt'
			torch.save(self.slmInputPlaneField, slmInputPlaneFieldPath)
			print("Saved SLM input plane field to %s" % (slmInputPlaneFieldPath.resolve()))

		if (self.save_data_directory is None):
			print("NOTE: Not saving SLM data.")
		else:
			self.slmModel.save_all_slms_into_folder(self.saveDataSLMDataFolder)
			
			metadataPath = self.saveDataFolder / 'Metadata.pt'
			metadataDict =	{
								'save_version' 				:	'1.0',								# Keeping track of the version in case the way data is saved changes in the future and we need to write code that is backwards compatible with old data.
								'datetime'					:	str(self.dataFolderCreationTime),
								'image_plane_type'			:	self.imagePlaneType,				# It might also be useful to know whether the image plane was on the image sensor or the SLM.
								'inputImageFilename'		:	self.inputImageFilepath,
								'objectDepthRange'			:	self.objectDepthRange,
								'slm_class_name'			: 	self.slmModel.__class__.__name__,	# Keeping track of the type of SLM as that might be relevant to know.
								'slmPlanePixelPitch'		:	self.slmPlanePixelPitch,
								'sensorPlanePixelPitch'		:	self.sensorPlanePixelPitch,
								'additional_data'			:	self.extraMetadataDictionary
							}
			torch.save(metadataDict, metadataPath)
			print("Saved metadata to %s" % (metadataPath.resolve()))

			creatorScriptSavePathStr = self.save_creator_script_to_directory()
			print("Saved metadata to %s" % (creatorScriptSavePathStr))

		for batch_id in range(n_batches):
			# Load in SLM batch data
			slmData = self.slmModel.load_single_slm(batch_idx=batch_id)
			sensorPlaneFieldTemp = self.slmInputToSensorOutputModel(self.slmInputPlaneField).detach()	# Calling detach() here allows some memory to be reclaimed.  Without it, memory would fill up.

			if (self.useFloat16ForIntensity):
				sensorPlaneFieldTemp.data = sensorPlaneFieldTemp.data.to(dtype=torch.float16)

			if (self.saveSensorMeasurementsInMemory):
				self.sensorMeasurements[batch_id] = sensorPlaneFieldTemp.cpu()

			# if (clearCudaCacheFlag):
			# 	torch.cuda.empty_cache() # Helps prevent the GPU memory from filling up
			# 	print_cuda_memory_usage(self.device)
			# 	print()

			if (self.save_data_directory is None):
				print("NOTE: Computed SLM and intensity patterns for mini-batch #%d but did not save the results." % (batch_id))
			else:
				saveFileName = "SENSOR_DATA_" + str(batch_id).zfill(4) + ".pt"
				saveFilePath = self.saveDataSensorDataFolder / saveFileName

				saveDataDict =	{
									'data' 			:	sensorPlaneFieldTemp.data,
									'wavelengths' 	:	sensorPlaneFieldTemp.wavelengths,	# Saving the wavelengths and spacing objects directly because it's convenient
									'spacing'		:	sensorPlaneFieldTemp.spacing,		# Python's pickling methods are not good for code that might change.  Hopefully, one would still be able to parse the wavelength and spacing object data even if those classes change.
									'identifier'	:	sensorPlaneFieldTemp.identifier
								}
				torch.save(saveDataDict, saveFilePath)

				print("Saved sensor data to %s" % (saveFilePath.resolve()))
		
		print("Finished saving data.")