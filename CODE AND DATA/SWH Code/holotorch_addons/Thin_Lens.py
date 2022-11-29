########################################################
# This file is a modified copy of the FT_Lens.py class in HoloTorch (HoloTorch: Copyright (c) 2022 Meta Platforms, Inc. and affiliates).
########################################################

import sys

import torch
import numpy as np
import warnings
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.utils import Dimensions
import holotorch.utils.Dimensions as Dimension
from holotorch.utils.Helper_Functions import *
from holotorch.utils.units import *

import holotorch_addons.HolotorchPatches as HolotorchPatches


class Thin_Lens(CGH_Component):

	def __init__(self,
				focal_length,
				sign_convention : HolotorchPatches.ENUM_PHASE_SIGN_CONVENTION = HolotorchPatches.ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE
				):
		"""
		This implements a thin lens by applying the phase shift described in Equation (6-10) of Goodman's Fourier optics book
		"""

		if (sign_convention != HolotorchPatches.ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE):
			raise Exception("Need to investigate whether or not the Fourier transform sign convention changes from e^{-j\omega t} for the forward transform if time phasors rotate counterclockwise.")

		super().__init__()

		self.sign_convention = sign_convention
		self.focal_length = focal_length



	def _init_dimensions(self):
		"""Creates coordinate system for detector
		"""        

	def __str__(self, ):
		"""
		Creates an output for the Material class.
		"""
		
		mystr = super().__str__()
		mystr += "\n-------------------------------------------------------------\n"
		mystr += "Focal Length: " + str(self.focal_length/mm) + " mm \n"
		mystr += "Padding: " + str(self.pad)

		return mystr
	
	def __repr__(self):
		return self.__str__()


	def create_lens_phase_shift_kernel(self,
		field : ElectricField,
			):
		# The code in this function was derived from the ASM_Prop.py class
		# in HoloTorch (HoloTorch: Copyright (c) 2022 Meta Platforms, Inc. and affiliates).

		# extract dx, dy spacing into T x C tensors
		spacing = field.spacing.data_tensor
		dx      = spacing[:,:,0]
		if spacing.shape[2] > 1:
			dy = spacing[:,:,1]
		else:
			dy = dx

		# extract the data tensor from the field
		wavelengths = field.wavelengths

		# Extract width and height dimensions in pixels
		nHeight = field.data.size()[-2]
		nWidth = field.data.size()[-1]

		# get the wavelengths data as a TxC tensor 
		new_shape       = wavelengths.tensor_dimension.get_new_shape(new_dim=Dimension.TC)        
		wavelengths_TC  = wavelengths.data_tensor.view(new_shape) # T x C
		# Expand wavelengths for H and W dimension
		wavelengths_TC  = wavelengths_TC[:,:,None,None]
			
		# do the same for the spacing
		dx_TC   = dx.expand(new_shape)
		dx_TC   = dx_TC[:,:,None,None] # Expand to H and W
		dy_TC   = dy.expand(new_shape)
		dy_TC   = dy_TC[:,:,None,None] # Expand to H and W

		xCoordsTemp = torch.linspace(-((nHeight - 1) // 2), (nHeight - 1) // 2, nHeight)
		yCoordsTemp = torch.linspace(-((nWidth - 1) // 2), (nWidth - 1) // 2, nWidth)
		xGridTemp, yGridTemp = torch.meshgrid(xCoordsTemp, yCoordsTemp)
		
		xGrid_TC = xGridTemp[None,None,:,:].to(field.data.device) * dx_TC
		yGrid_TC = yGridTemp[None,None,:,:].to(field.data.device) * dy_TC

		ang = -(np.pi / (wavelengths_TC * self.focal_length)) * ((xGrid_TC ** 2) + (yGrid_TC ** 2))

		# Adjust angle to match sign convention
		#	For more information, see Section 4.2.1 in "Introduction to Fourier Optics" (3rd Edition) by Joseph W. Goodman
		if (self.sign_convention == HolotorchPatches.ENUM_PHASE_SIGN_CONVENTION.TIME_PHASORS_ROTATE_CLOCKWISE):
			# Do nothing---the calculations for 'ang' should have assumed this convention.
			pass
		else:
			ang = -ang

		ker = torch.exp(1j * ang)

		return ker


	def forward(self,
				field : ElectricField
				) -> ElectricField:
		"""
		In this function we apply a phase delay that simulates a thin lens

		Args:
			field(torch.complex128) : Complex field (MxN).
		"""
		# extract the data tensor from the field
		wavelengths = field.wavelengths
		field_data  = field.data
		
		# convert field to 4D tensor for batch processing
		B,T,P,C,H,W = field_data.shape
		field_data = field_data.view(B*T*P,C,H,W)

		phase_shift_ker = self.create_lens_phase_shift_kernel(field)

		field_data = field_data * phase_shift_ker[None,:,None,:,:,:]

		# convert field back to 6D tensor
		field_data = field_data.view(B,T,P,C,H,W)

		field.spacing.set_spacing_center_wavelengths(field.spacing.data_tensor)

		Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing = field.spacing
		)
		
		return Eout
