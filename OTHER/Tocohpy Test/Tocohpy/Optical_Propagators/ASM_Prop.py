

##########
#### This Package contains functions to propagate optical fields
#### Angular spectrum method with bandlimit 
####Author: Lionel Fiske 
####Last update 9/15/2021
#####

import torch
import numpy as np
import math    
import warnings
from Helper_Functions import * 



class ASM_Prop(torch.nn.Module):
	
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
		
		
		super(ASM_Prop, self).__init__()
		
		self.wavelength = wavelength
		self.dx = dx
		self.distance = distance
		self.device = device
		self.padding = padding

	#Compute H if not precomputed
		if H == None:
		#Max Frequency
			k =  2*np.pi/self.wavelength

			# Get the shape for processing
			Nx =int( (1 + 2*padding) * N)
			Ny =int( (1 + 2*padding) * N)

			kx_max =  .5/np.abs(self.dx)
			ky_max = kx_max

			# OLD CODE - Compute 1D-grid for both directions
				# This code seems to give slightly incorrect kx and ky values as the values are not aligned with the FFT bin frequencies:
					# kx = 2*np.pi*torch.linspace(-kx_max,kx_max,Nx, device = self.device);
					# ky = 2*np.pi*torch.linspace(-ky_max,ky_max,Ny, device = self.device);

			# Compute 1D-grid for both directions
			kx = 2*np.pi*(2*kx_max/Nx)*np.linspace(-np.floor(Nx/2), np.ceil(Nx/2) - 1, Nx)
			ky = 2*np.pi*(2*ky_max/Ny)*np.linspace(-np.floor(Ny/2), np.ceil(Ny/2) - 1, Ny)

			# Compute the meshgrid to represent Kx Ky coordinates
			Kx,Ky = torch.meshgrid(kx, ky)

			# Compute the ASM kernel
			ang = self.distance*torch.sqrt(  (k**2 - Kx**2 - Ky**2 )  ).type(dtype)
			H =  torch.exp(1j*ang)

			# zero out the invalid portion according to the circ function in Eq. 4.22
			invalid = (Kx**2 + Ky**2) > k**2
			H[invalid] = 0


		self.H = torch.unsqueeze(torch.unsqueeze(  H , dim=0 ) ,dim=0) .to(device)
	
		
	def forward(self, field):
		"""
		Takes in optical field and propagates it to the instantiated distance using ASM from KIM
		Eq. 4.22 (page 50)

		
		Parameters
		==========
		field            : torch.complex128
						   Complex field (MxN).
					   
		"""
		

		#Grab size
		Nx =  field.shape[-2]
		Ny =  field.shape[-1]

		# Save for later crop
		Nx_old = int(Nx)
		Ny_old = int(Ny)

		# Pad the image for avoiding convolution artifacts
		Ein = torch.nn.functional.pad(field, (int( self.padding * Nx_old),int(self.padding *Nx_old),int(self.padding *Ny_old),int(self.padding*Ny_old)), mode='constant', value=0)

		
		# Apply the ASM kernel
		Ein_fft = ft2( Ein )  
		Eout_no_pad =ift2(Ein_fft*self.H) 
		
		
		# Unpad the image the original size
			# Changed the last two N*_old to Ny_old.  Was Nx_old before, but that seemed like a mistake.
		Eout = torch.nn.functional.pad(Eout_no_pad, (-int(self.padding *Nx_old),-int(self.padding *Nx_old),-int(self.padding *Ny_old),-int(self.padding *Ny_old)), mode='constant', value=0)

		if field.ndim == 2:
			return Eout.squeeze()
		else:
			return Eout 
		return Eout 

####
