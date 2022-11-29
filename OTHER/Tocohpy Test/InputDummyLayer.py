##########
#### This Package contains components to build optical setups. 
#### SLM / Phase grating class. This component applies a learnable pointwise phase delay to the field
####Author: Lionel Fiske 
####Last update 9/15/2021
#####


import torch
import numpy as np
import math    
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 



class InputLDummyLayer(torch.nn.Module):
	
	def __init__(self, weights,  device = torch.device("cpu"), dtype=torch.complex128, fixed_weights = True ):
		"""
		Applies phase delay to the incident field. 
		
		Parameters
		==========
		weights		       : float Tensor
							Weights

						   
		device             :  torch device
		
		
		fixed_weights      : Bool
						   If True the phase delay will not be set to an nn.parameter to be optimized for 
				   
		"""
		
		super().__init__()
		
		
		# Check if parameter
		if fixed_weights == False:
			self.weights = torch.nn.Parameter(weights.to(device).type(dtype),requires_grad=True)
		else :
			self.weights = weights.to(device).type(dtype)

		
		self.device = device
		self.fixed_weights = fixed_weights
		self.dtype = dtype

		
	def forward(self, input):
		"""
		Takes in complex tensor and does pointwise multiplication
		
		Inputs
		==========
		input            : torch.complex128
						   Complex input (MxN).
		"""
		
		output = input * self.weights


		if input.ndim == 2:
			return output.squeeze()
		else:
			return output 

		
