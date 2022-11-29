# from logging import exception
# import wave
import numpy as np
import sys
import os
import warnings
import torch
import torchvision
from torch.nn.functional import pad
import matplotlib.pyplot as plt

import glob
import re
from pathlib import Path

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.utils.Enumerators import *
import holotorch.utils.Dimensions as Dimensions
from holotorch.HolographicComponents.Modulator_Container import Modulator_Container
from holotorch.HolographicComponents.SLM_PhaseOnly import SLM_PhaseOnly
from holotorch.HolographicComponents.ValueContainer import ValueContainer
from holotorch.utils.Dimensions import TensorDimension
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
from holotorch.utils.Enumerators import *
from holotorch.utils.Helper_Functions import ft2, ift2

########################################################################################################################
#### TODO SECTION                                                                                                   ####
########################################################################################################################
# TODO: For Modulator_Container, test set_new_values(...) for store_on_gpu = False case.
# TODO: For Modulator_Container, implement set_new_values(...) for store_on_gpu = True case.
########################################################################################################################




# def ElectricField_visualize_patched(	self,
# 										title: str = "",
# 										flag_colorbar: bool = True,
# 										flag_axis: str = False,
# 										cmap='gray',
# 										index=None,
# 										open_new_figure=False,
# 										figsize=None,
# 										vmax=None,
# 										vmin=None,
# 										plot_type : ENUM_PLOT_TYPE = ENUM_PLOT_TYPE.MAGNITUDE,
# 										flag_log : bool = False,
# 										adjust_aspect : bool    = False,
# 										rescale_factor : float = 1,
# 									):

# 		if plot_type is ENUM_PLOT_TYPE.MAGNITUDE:
# 			new_intensity_field = self.get_intensity()
# 		elif plot_type is ENUM_PLOT_TYPE.PHASE:
# 			new_intensity_field = self.angle()

# 		if flag_log == True:
# 			new_intensity_field = new_intensity_field.log()

# 		new_intensity_field.visualize(
# 			title = title,
# 			flag_colorbar = flag_colorbar,
# 			flag_axis = flag_axis,
# 			cmap = cmap,
# 			index = index,
# 			open_new_figure = open_new_figure,
# 			figsize = figsize,
# 			vmax = vmax,
# 			vmin = vmin,
# 			adjust_aspect = adjust_aspect,
# 			rescale_factor = rescale_factor,
# 		)


# def doMonkeyPatch():
# 	ElectricField.visualize = ElectricField_visualize_patched
# 	print("")
# 	print("#### NOTE ####")
# 	print("  Monkey patching was performed in HolotorchPatches.py by the doMonkeyPatch() method.")
# 	print("  Some class functions were dynamically altered.")

# doMonkeyPatch()








class ENUM_PHASE_SIGN_CONVENTION(Enum):
	TIME_PHASORS_ROTATE_CLOCKWISE = 1
	TIME_PHASORS_ROTATE_COUNTERCLOCKWISE = -1


class Modulator_Container_Patched(Modulator_Container):
	def __init__(	self,
					tensor_dimension : TensorDimension,
					feature_size    : float,
					n_slm_batches               = 1,
					replicas :int               = 1,
					pixel_fill_ratio: float     = 1.0,
					pixel_fill_ratio_opt: bool  = False,
					init_type       : ENUM_SLM_INIT = None,
					init_variance : float = 0,
					flag_complex : bool         = False,
					slm_directory : str         = ".slm",
					slm_id : int                = 0,
					store_on_gpu                = False,
				) -> None:

		if ((tensor_dimension.batch % n_slm_batches) != 0):
			raise Exception("Error: 'n_slm_batches' does not evenly divide the number of batches.")

		if not (hasattr(self, 'static_slm')):
			raise Exception("A subclass failed to set the 'static_slm' attribute.")
		if not (hasattr(self, 'static_slm_data_path')):
			raise Exception("A subclass failed to set the 'static_slm_data_path' attribute.")

		if (self.static_slm == False):
			if (self.static_slm_data_path is not None):
				raise Exception("static_slm_data_path' is set, but static_slm is False.")
		else:
			if (self.static_slm_data_path is None):
				raise Exception("'static_slm' is set to True, but 'static_slm_data_path' is not set.")
			elif (not issubclass(type(self.static_slm_data_path), Path)):
				raise Exception("'static_slm_data_path' must be a pathlib.Path object.")
			elif not (self.static_slm_data_path.exists()):
				raise Exception("The pathlib.Path object provided for 'static_slm_data_path' does not point to a valid path.")

		self.input_arg_init_variance = init_variance
		self.input_arg_init_type = init_type
		self.input_arg_flag_complex = flag_complex

		super().__init__(	tensor_dimension = tensor_dimension,
							feature_size = feature_size,
							n_slm_batches = n_slm_batches,
							replicas = replicas,
							pixel_fill_ratio = pixel_fill_ratio,
							pixel_fill_ratio_opt = pixel_fill_ratio_opt,
							init_type = init_type,
							init_variance = init_variance,
							flag_complex = flag_complex,
							slm_directory = slm_directory,
							slm_id = slm_id,
							store_on_gpu = store_on_gpu
						)

		if (self.static_slm):
			self.load_all_slms_from_folder(self.static_slm_data_path)

	def init_on_gpu(
				self,
				batch_tensor_dimension  : TensorDimension,
				init_variance           : float,
				init_type               : ENUM_SLM_INIT,
				flag_complex            : bool,
				slm_directory,
				slm_id,
				images_per_batch,
				n_slm_batches
		):

		for k in range(n_slm_batches):
			tmp_values : ValueContainer = ValueContainer(
				tensor_dimension = batch_tensor_dimension,
				init_variance    = init_variance,
				init_type        = init_type,
				flag_complex     = flag_complex,
			)
			if not (self.device is None):
				tmp_values.data_tensor = tmp_values.data_tensor.to(self.device)
				tmp_values.scale = tmp_values.scale.to(self.device)
			setattr(self,"slm" + str(k),tmp_values)

		self.n_slm_batches = n_slm_batches
		self.current_batch_idx = 0


	def init_on_disk(self,
		batch_tensor_dimension  : TensorDimension,
		init_variance           : float,
		init_type               : ENUM_SLM_INIT,
		flag_complex            : bool,
		slm_directory,
		slm_id,
		images_per_batch,
		n_slm_batches
					):

		# Create a directory where temporary SLM data is stored.  Initializes self.tmp_dir
		self.create_tmp_directory(tmp_name=slm_directory, slm_id = slm_id)

		# Clear out files in the temporary SLM directory (that end in .pt)
		self.clear_temp_slm_dir()

		# Initializing a values container (needs to be initialized for self.set_images_per_batch(...) to work)
		self.values : ValueContainer = ValueContainer(
			tensor_dimension = batch_tensor_dimension,
			init_variance    = init_variance,
			init_type        = init_type,
			flag_complex     = flag_complex,
		)

		# Initializing field
		self.n_slm_batches = n_slm_batches

		for k in range(n_slm_batches - 1, -1, -1):
			with torch.no_grad():
				# Updating batch-related fields in values container and generating new SLM data (which is contained in self.values.data_tensor).
				self.values.set_images_per_batch(number_images_per_batch=images_per_batch)

				# NOTE that the self.values.data_tensor and self.values.scale tensors would have been computed on the CPU and will reside on the CPU at this point.
				#	(Unless something changes after 9/8/2022)

				# Saving the SLM data
				self.save_single_slm(batch_idx=k)

		# Moving tensors to a different device if necessary
		if not (self.device is None):
			self.values.to(self.device)

		# Forcing the currently loaded SLM to be idx=0
		self.current_batch_idx = 0
		self.load_single_slm(batch_idx=0)	# This line is technically not necessary because 'k' ends up as 0 in the loop 'for k in range(n_slm_batches - 1, -1, -1)'


	def init_save_directory(self,
							tmp_name : str,
							slm_id   : int,
							images_per_batch : int,
							n_slm_batches : int,
						):
		raise Exception("The method 'init_save_directory' is not used in the implementation of 'Modulator_Container_Patched'.")


	# NOTE: Actually, this seems to be used.  Maybe remove this?
	def set_images_per_batch(self, number_images_per_batch : int, number_slm_batches = None):
		raise Exception("The method 'set_images_per_batch' is not used in the implementation of 'Modulator_Container_Patched'.")


	# Should hopefully not be necessary:
	# @property
	# def values(self):
	# 	if (self.store_on_gpu):
	# 		return self.load_single_slm(batch_idx=self.current_batch_idx)
	# 	return self.values


	def save_single_slm(self,
						batch_idx :int  = None,
						folder : str = None,
						filename : str = None
						):
		if (hasattr(self, 'current_batch_idx')):
			if batch_idx == None:
				batch_idx = self.current_batch_idx
			if ((batch_idx != self.current_batch_idx) and (not self.store_on_gpu)):
				self.load_single_slm(batch_idx=batch_idx)
		# else:
		# 	# Reaching this point means class is still initializing
		# 	pass

		slm_path = self._create_file_path(batch_idx = batch_idx, folder = folder, filename=filename)

		if (self.store_on_gpu):
			tempValues = self.load_single_slm(batch_idx=batch_idx)
			state_dict = tempValues.state_dict()
		else:
			# If 'current_batch_idx' was not set, then the class is still initializing.  This means that whatever is stored in self.values are the initialized values, which means that they should just be saved.
			# If 'current_batch_idx' was set, then either the correct values were already loaded before this method was called, or the correct values would have been loaded earlier in this method.
			# Therefore, one can just have this single line.
			state_dict = self.values.state_dict()

		# If not static_slm, save.  If folder is not none, then assume that the user is explicitly trying to save an SLM, rather than the code trying to save a temporary SLM file.
		if (not self.static_slm) or (folder is not None):
			torch.save(state_dict, slm_path)


	def load_single_slm(	self,
							batch_idx,
							folder : str = None,
							filename : str = None,
							flag_save_current = True,
						):
		if ((filename is not None) or (folder is not None)):
			raise Exception("Use case not covered.")
		else:
			if self.store_on_gpu:
				load_function = self.load_single_slm_gpu		# Nothing different needs to be done
			elif not self.static_slm:
				load_function = self.load_single_slm_disk
			else:
				folder = self.static_slm_data_path.resolve()		# The call to resolve() is technically not needed.  Even though self.static_slm_data_path should be a pathlib.Path object and not a string, the functions that argument gets passed to should be able to handle a pathlib.Path object.
				load_function = self.load_single_slm_disk_static

			slm = load_function(
									batch_idx = batch_idx,
									folder=folder,
									filename=filename,
									flag_save_current = flag_save_current
								)


			return slm.to(self.device)


	def load_single_slm_gpu(	self,
								batch_idx,
								folder : str = None,
								filename : str = None,
								flag_save_current = True,
							):
		self.current_batch_idx = batch_idx
		slm = super().load_single_slm_gpu(
			batch_idx = batch_idx,
			folder = folder,
			filename = filename,
			flag_save_current = flag_save_current
		)
		return slm


	# Loads an SLM without saving anything to disk
	def load_single_slm_disk_static(self,
									batch_idx,
									folder : str = None,
									filename : str = None,
									flag_save_current = False	# This argument does nothing in this function and only serves to make it compatible with other function calls
									):
		if batch_idx == self.current_batch_idx:
			# We don't need to do anything if the current batch is already loaded
			return self.values

		self.current_batch_idx = batch_idx
		slm_path = self._create_file_path(batch_idx = batch_idx, folder = folder, filename=filename)

		# there is only one slm object
		new_state_dict = torch.load(slm_path)
		self.values.load_state_dict(new_state_dict)

		return self.values


	def save_all_slms_into_folder(self, folder):
		self.clear_save_dir(folder)
		super().save_all_slms_into_folder(folder)


	def load_all_slms_from_folder(self, folder):
		"""
		load slm state_dicts from a specific folder

		Args:
			slmmodel_folder: the folder where all SLM state_dicts are stored
		"""

		# Clear out files in the temporary SLM directory (that end in .pt)
		self.clear_temp_slm_dir()

		files = glob.glob(str(folder)+"\\*.pt")
		
		n_batches = len(files)
		self.n_slm_batches = n_batches

		batch_size = -1
		n_batches = -1

		for k in range(len(files)):
			file = Path(files[k])
			filename = str(file.stem)
			# Finds the index
			idx_slm = [int(x) for x in re.findall('\d+', filename)][0]
			slm_state_dict = torch.load(file, map_location=self.device)

			self.current_batch_idx = idx_slm
			slmName = "slm" + str(idx_slm)
			
			newContainerDimShape = slm_state_dict['_data_tensor'].shape
			if (len(newContainerDimShape) == 5):
				newContainerDim = Dimensions.BTCHW(
					n_batch     = newContainerDimShape[0], # Total number of images for Modualator_Container
					n_time      = newContainerDimShape[1],
					n_channel   = newContainerDimShape[2],
					height      = newContainerDimShape[3],
					width       = newContainerDimShape[4]
				)
			else:
				raise Exception("SLM data tensors should have dimensions BTCHW.")

			if k == 0:
				batch_size = slm_state_dict['_data_tensor'].shape[0]

				with torch.no_grad():
					if not (self.store_on_gpu):
						self.values : ValueContainer = ValueContainer(	tensor_dimension = newContainerDim,
																		init_variance    = self.input_arg_init_variance,
																		init_type        = self.input_arg_init_type,
																		flag_complex     = self.input_arg_flag_complex,
																	)

			if not (self.store_on_gpu):
				self.values.load_state_dict(slm_state_dict)
				if not (self.static_slm):
					self.save_single_slm(batch_idx=idx_slm)
			else:
				tmp_values : ValueContainer = ValueContainer(
					tensor_dimension = newContainerDim,
					init_variance    = self.input_arg_init_variance,
					init_type        = self.input_arg_init_type,
					flag_complex     = self.input_arg_flag_complex,
				)
				setattr(self, slmName, tmp_values)

				with torch.no_grad():
					getattr(self, slmName).set_images_per_batch(number_images_per_batch=batch_size)
				getattr(self, slmName).load_state_dict(slm_state_dict)

				# Moving data to proper device if necessary
				if not (self.device is None):
					getattr(self, slmName).data_tensor = getattr(self, slmName).data_tensor.to(self.device)
					getattr(self, slmName).scale = getattr(self, slmName).scale.to(self.device)


	def clear_temp_slm_dir(self):
		# Removes temporary stored data
		if not (hasattr(self, 'tmp_dir')):
			warnings.warn("Tried to clear the temporary SLM data directory, but the corresponding field 'tmp_dir' was uninitialized.")
			return
		elif (self.tmp_dir is None):
			warnings.warn("Tried to clear the temporary SLM data directory, but the corresponding field 'tmp_dir' was set to None.")
			return

		test = os.listdir(self.tmp_dir)
		for item in test:
			if item.endswith(".pt"):
				os.remove(os.path.join(self.tmp_dir, item))


	def clear_save_dir(self, folder):
		# Removes saved data
		if (Path(folder).exists()):
			test = os.listdir(folder)
			for item in test:
				if item.endswith(".pt"):
					os.remove(os.path.join(folder, item))





class SLM_PhaseOnly_Patched(SLM_PhaseOnly, Modulator_Container_Patched):
	def __init__(	self,
					tensor_dimension 		: TensorDimension,
					feature_size			: float,
					n_slm_batches			= 1,
					replicas				: int = 1,
					pixel_fill_ratio		: float = 1.0,
					pixel_fill_ratio_opt	: bool  = False,
					init_type				: ENUM_SLM_INIT = None,
					init_variance			: float = 0,
					FLAG_optimize			: bool = True,
					slm_directory			: str = ".slm",
					slm_id					: int = 0,
					static_slm				= False,
					static_slm_data_path	= None,
					store_on_gpu			= False,
					device					: torch.device = None,
				):

		self.static_slm = static_slm
		self.static_slm_data_path = static_slm_data_path
		self.device = device

		super().__init__(
			tensor_dimension            = tensor_dimension,
			feature_size                = feature_size,
			n_slm_batches               = n_slm_batches,
			replicas                    = replicas,
			pixel_fill_ratio            = pixel_fill_ratio,
			pixel_fill_ratio_opt        = pixel_fill_ratio_opt,
			init_type                   = init_type,
			init_variance               = init_variance,
			FLAG_optimize   			= FLAG_optimize,	# This parameter does not seem to do anything (at least as of 9/2/2022).
															# The SLM_PhaseOnly class has FLAG_optimize as an argument to its initializer, but does not appear to use it.
			slm_directory               = slm_directory,
			slm_id                      = slm_id,
			store_on_gpu                = store_on_gpu,
		)


	@classmethod
	def create_slm(	cls,
					height					: int,
					width					: int,
					feature_size			: float,
					replicas				: int = 1,
					pixel_fill_ratio		: float = 1.0,
					pixel_fill_ratio_opt	: bool = False,
					init_type       		: ENUM_SLM_INIT = None,
					init_variance   		: float = 0,
					FLAG_optimize   		: bool = True,	# This parameter does not seem to do anything (at least as of 9/2/2022).
															# The SLM_PhaseOnly class has FLAG_optimize as an argument to its initializer, but does not appear to use it.
					n_batch  				: int = 1, # Total number of images for Modualator_Container
					n_time   				: int = 1,
					n_channel 				: int = 1,
					n_slm_batches			: int = 1,
					slm_directory			: str = '.slm',
					static_slm				: bool = False,
					static_slm_data_path	: Path = None,
					store_on_gpu            : bool = False,
					device					: torch.device = None,
				) -> SLM_PhaseOnly:

		SLM_container_dimension = Dimensions.BTCHW(
			n_batch         = n_batch, # Total number of images for Modualator_Container
			n_time          = n_time,
			n_channel       = n_channel,
			height          = height,
			width           = width
		)

		return SLM_PhaseOnly_Patched(
			tensor_dimension = SLM_container_dimension,
			n_slm_batches = n_slm_batches,
			feature_size    = feature_size,
			replicas  = replicas,
			pixel_fill_ratio = pixel_fill_ratio,
			pixel_fill_ratio_opt = pixel_fill_ratio_opt,
			init_type     = init_type,
			init_variance  = init_variance,
			FLAG_optimize  = FLAG_optimize,
			slm_directory = slm_directory,
			static_slm = static_slm,
			static_slm_data_path = static_slm_data_path,
			store_on_gpu = store_on_gpu,
			device = device
		)


	# This method updates the field.wavelengths.tensor_dimension field to better keep track of what each dimension in field.data represents (e.g. B, T, P, C, H, W).
	#					- (The SLM_PhaseOnly superclass's forward method can sometimes expand the batch dimension to be of size > 1, without updating the dimensions of field.wavelength.tensor_dimension)
	# Doing this is inconsequential if one already knows that information (e.g. some methods automatically return data in BTPCHW form).
	# However, in specific cases, some ambiguities can arise.
	# def forward(self,
	# 			field : ElectricField = None,
	# 			batch_idx = None,
	# 			bit_depth : int = None,
	# 		) -> ElectricField:

	# 	# Note that the SLM_PhaseOnly superclass will change the batch dimension size to equal the number of images per batch.
	# 	unpatchedField = super().forward(field=field, batch_idx=batch_idx, bit_depth=bit_depth)

	# 	if (torch.is_tensor(unpatchedField)):
	# 		return unpatchedField

	# 	dataOut = unpatchedField.data
	# 	if (dataOut.device != self.device):
	# 		dataOut = dataOut.to(device=self.device)

	# 	# Since field.data does not keep track of dimension labels itself, using
	# 	# the wavelength container to keep track of it.
	# 	wavelengthDataOutShape = unpatchedField.wavelengths.tensor_dimension.get_new_shape(Dimensions.BTPC)
	# 	wavelengthDataOut = unpatchedField.wavelengths.data_tensor.view(wavelengthDataOutShape)
	# 	wavelengthsOut = WavelengthContainer(
	# 		wavelengths = wavelengthDataOut,
	# 		tensor_dimension =	Dimensions.BTPC(	n_batch=wavelengthDataOutShape[0],
	# 												n_time=wavelengthDataOutShape[1],
	# 												n_pupil=wavelengthDataOutShape[2],
	# 												n_channel=wavelengthDataOutShape[3],
	# 										)
	# 	)
	# 	if (wavelengthDataOut.device != self.device):
	# 		wavelengthsOut = wavelengthsOut.to(device=self.device)

	# 	# Not bothering to alter the spacing container's dimensions because
	# 	# other parts of the code (e.g. ASM_Prop.py and possibly other parts as well)
	# 	# assume TCD dimensions for spacing.  If the spacing was set to have BTPCH dimensions, then methods
	# 	# that get dx in a manner similar to 'dx = spacing_container.data[:,:,0]' (e.g. create_kernel(...) in ASM_Prop.py)
	# 	# will not behave correctly.
	# 	if (unpatchedField.spacing.data_tensor.device != self.device):
	# 		spacingOut = unpatchedField.spacing.to(self.device)
	# 	else:
	# 		spacingOut = unpatchedField.spacing
	# 	spacingOut.set_spacing_center_wavelengths(spacingOut.data_tensor)

	# 	Eout = ElectricField(
	# 		data = dataOut,
	# 		wavelengths = wavelengthsOut,
	# 		spacing = spacingOut
	# 	)

	# 	return Eout