import numpy as np
import sys
import torch
from torch.nn.functional import pad
import matplotlib.pyplot as plt

import copy

from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.Optical_Components.CGH_Component import CGH_Component
import holotorch.utils.Dimensions as Dimensions
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.utils.Enumerators import *

from holotorch_addons.HelperFunctions import applyFilterToElectricField, get_field_slice, print_cuda_memory_usage, get_tensor_size_bytes, generateGrid, fit_image_to_resolution
from holotorch_addons.Field_Resampler import Field_Resampler


# This class allows for propagation between two SQUARE-SHAPED planes of different sides and the same number of samples.
#	- SOURCE: Based off of the Fresnel two-step propagator described in Appendix of "Computational Fourier Optics: A MATLAB Tutorial" by David Voelz.
#				MATLAB code in that reference was adapted for Python/PyTorch.
class Fresnel_Two_Step_Prop(CGH_Component):
	def __init__(	self,
					M							: int,
					delta1						: float,
					delta2						: float,
					propagationDistance			: float,
					resampleAtInput				: bool = True,
					device						: torch.device = None,
					gpu_no						: int = 0,
					use_cuda					: bool = False
				) -> None:
		
		super().__init__()

		if (device != None):
			self.device = device
		else:
			self.device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
			self.gpu_no = gpu_no
			self.use_cuda = use_cuda

		if (type(M) is not int) or (M <= 0):
			raise Exception("Bad argument: 'M' should be a positive integer.")
		if (type(delta1) is not float) or (delta1 <= 0):
			raise Exception("Bad argument: 'delta1' should be a positive real number.")
		if (type(delta2) is not float) or (delta2 <= 0):
			raise Exception("Bad argument: 'delta1' should be a positive real number.")
		if (type(propagationDistance) is not float) or (propagationDistance <= 0):
			raise Exception("Bad argument: 'propagationDistance' should be a positive real number.")

		self.M = M
		self.dx1 = delta1
		self.dx2 = delta2
		self.z = propagationDistance

		self.l1 = M * self.dx1
		self.l2 = M * self.dx2


		# Want to have centerAroundZero=False so that there is a point on the grid that corresponds to the point (x,y) = (0,0),
		#	AND that said point gets mapped to index [0][0] when fftshift(...) is called.  The reason for this is that PyTorch's
		#	fft2 method views the index [0][0] as the (0,0) coordinate.
		self.xGrid1, self.yGrid1	= generateGrid((M,M), self.dx1, self.dx1, centerGrids=True, centerAroundZero=False, device=device)
		self.xGrid2, self.yGrid2	= generateGrid((M,M), self.dx2, self.dx2, centerGrids=True, centerAroundZero=False, device=device)

		self.fxGrid1, self.fyGrid1	= generateGrid((M,M), 1/self.l1, 1/self.l1, centerGrids=True, centerAroundZero=False, device=device)


		if (resampleAtInput):
			# This will be used to resample inputs to match the dimensions specified
			self.inputResampler =	Field_Resampler(	outputHeight = M,
														outputWidth = M,
														outputPixel_dx = self.dx1,
														outputPixel_dy = self.dx1,
														device = device
													)
		self.resampleAtInput = resampleAtInput
	
	
	def forward(self, fieldIn):
		if (self.resampleAtInput):
			# Resample the field to fit the dimensions specified
			field = self.inputResampler(fieldIn)
		else:
			spacingMismatchFlag = False
			epsilon = 1e-12
			if (fieldIn.spacing.data_tensor.numel() == 0):
				spacingMismatchFlag = True
			elif (torch.any((fieldIn.spacing.data_tensor - self.dx1).abs() > epsilon)):
				spacingMismatchFlag = True
			if (spacingMismatchFlag):
				raise Exception("Mismatch between specified input sample spacing and the spacing given in the input field.  Either correct the spacing or set resampleAtInput to True.")
			field = fieldIn

		# Reshape the data to 4D tensor for batch processing
		Bf,Tf,Pf,Cf,Hf,Wf = field.data.shape
		u1 = field.data.view(Bf*Tf*Pf,Cf,Hf,Wf)

		# Get the wavelengths data as a TxC tensor, and then expand it to a TCHW tensor
		wavelengths 	= field.wavelengths
		new_shape       = wavelengths.tensor_dimension.get_new_shape(new_dim=Dimensions.TC)
		wavelengths_TC  = wavelengths.data_tensor.view(new_shape) # T x C
		wavelengths_TC  = wavelengths_TC[:,:,None,None] # Expand wavelengths for H and W dimension

		# Calculate wavenumbers
		k = 2*np.pi / wavelengths_TC

		dx1 = self.dx1
		dx2 = self.dx2
		l1 = self.l1
		l2 = self.l2
		z = self.z
		xGrid1 = self.xGrid1
		yGrid1 = self.yGrid1
		fxGrid1 = self.fxGrid1
		fyGrid1 = self.fyGrid1
		xGrid2 = self.xGrid2
		yGrid2 = self.yGrid2

		# Source plane
		u = u1 * torch.exp(1j * k/(2*z*l1) * (l1 - l2) * ((xGrid1**2) + (yGrid1**2)))
		u = torch.fft.fft2(torch.fft.fftshift(u))	# fftshift(u) makes sure the zero coordinate is at index [0][0].  This is needed because the FFT expects the zero coordinate to be in that location.

		# Dummy (frequency) plane
		u = u * torch.exp(-1j*np.pi * wavelengths_TC * z * (l1/l2) * ((torch.fft.fftshift(fxGrid1)**2) + (torch.fft.fftshift(fyGrid1)**2)))
		u = torch.fft.ifftshift(torch.fft.ifft2(u))

		# Observation plane
		u2 = (l2/l1) * u * torch.exp(-1j*k / (2*z*l2) * (l1 - l2) * ((xGrid2**2) + (yGrid2**2)))
		u2 = u2 * (dx1**2) / (dx2**2)	# Adjusting for differing scales between the planes

		u2 = u2.view(Bf,Tf,Pf,Cf,Hf,Wf)

		Eout = 	ElectricField(
					data = u2,
					wavelengths = field.wavelengths,
					spacing = dx2
				)
		Eout.wavelengths = Eout.wavelengths.to(fieldIn.data.device)
		Eout.spacing = Eout.spacing.to(fieldIn.data.device)

		return Eout