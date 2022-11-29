##########
#### This Package contains functions to propagate optical fields
#### Angular spectrum method with bandlimit 
####Author: Lionel Fiske 
####Last update 9/15/2021
#####

import sys

import torch
import numpy as np
import math    
import warnings

import matplotlib.pyplot as plt

import Tocohpy.Optical_Propagators as prop
from Helper_Functions import * 



class ASM_Prop_With_Backprop(prop.ASM_Prop):
	
	def __init__(self, wavelength, dx, distance , N , H=None, padding = 1/2 , dtype =torch.complex128 ,device = torch.device("cpu")):
		"""
		Angular Spectrum method with bandlimited ASM from Digital Holographic Microscopy
		Principles, Techniques, and Applications by K. Kim 
		Eq. 4.22 (page 50)
		
		Parameters
		==========
						   
		wavelength        : float
						   Wavelength of light
		
		dx                : float
						   Pixel Size (same units as wavelength)
						   
		distance          : float
						  Propagation Distance
	   
		N                  : int 
						  Simulation size
		
									 
		H                 : torch.complex128
						  Simulation size
						 
		padding           : float
						  percent of domain to pad for simulation
										   
						 
								 
		device             :  torch device
						   
		"""
		
		
		super().__init__(wavelength, dx, distance, N, H = H, padding = padding, dtype = dtype, device = device)
		self.inverseH = torch.conj(self.H) # Elements of self.H should either be 0 or a complex number of magnitude 1.  Thus, self.inverseH should be the complex conjugate of self.H to cancel out (most of) its effects.


	def propagate(self, field, propagationArray, dataIsPrePadded = False, doUnpadding = True):
		if dataIsPrePadded:
			Nx =  field.shape[-2]
			Ny =  field.shape[-1]
			Nx_old = int(Nx / (2*self.padding + 1))
			Ny_old = int(Ny / (2*self.padding + 1))
			
			Ein = field
		else:
			Nx_old = int(field.shape[-2])
			Ny_old = int(field.shape[-1])

			# Pad the image for avoiding convolution artifacts
			Ein = torch.nn.functional.pad(field, (int( self.padding * Nx_old),int(self.padding *Nx_old),int(self.padding *Ny_old),int(self.padding*Ny_old)), mode='constant', value=0)

		# Apply the ASM kernel
		Ein_fft = ft2( Ein )
		Eout_padded =ift2(Ein_fft*propagationArray)

		if doUnpadding:
			# Unpad the image the original size
				# Changed the last two N*_old to Ny_old.  Was Nx_old before, but that seemed like a mistake.
			Eout = torch.nn.functional.pad(Eout_padded, (-int(self.padding *Nx_old),-int(self.padding *Nx_old),-int(self.padding *Ny_old),-int(self.padding *Ny_old)), mode='constant', value=0)
		else:
			Eout = Eout_padded

		if field.ndim == 2:
			return Eout.squeeze()
		else:
			return Eout


	def forwardPropagate(self, field, dataIsPrePadded = False, doUnpadding = True):
		return self.propagate(field, self.H, dataIsPrePadded = dataIsPrePadded, doUnpadding = doUnpadding)


	def invertForwardPropagation(self, field, dataIsPrePadded = False, doUnpadding = True):
		return self.propagate(field, self.inverseH, dataIsPrePadded = dataIsPrePadded, doUnpadding = doUnpadding)

		
	def forward(self, field):
		"""
		Takes in optical field and propagates it to the instantiated distance using ASM from KIM
		Eq. 4.22 (page 50)

		
		Parameters
		==========
		field            : torch.complex128
						   Complex field (MxN).
					   
		"""
		return self.forwardPropagate(field)

####
