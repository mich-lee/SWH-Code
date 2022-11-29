import numpy as np
import sys
import torch
from torch.nn.functional import grid_sample
import matplotlib.pyplot as plt

from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.Optical_Components.CGH_Component import CGH_Component
import holotorch.utils.Dimensions as Dimension
from holotorch.CGH_Datatypes.ElectricField import ElectricField

from HelperFunctions import generateGrid


class Field_Resampler(CGH_Component):
	def __init__(	self,
					outputHeight						: int,
					outputWidth							: int,
					outputPixel_dx						: float,
					outputPixel_dy						: float,
					device								: torch.device = None,
					gpu_no								: int = 0,
					use_cuda							: bool = False
				) -> None:
		
		super().__init__()

		if (device != None):
			self.device = device
		else:
			self.device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
			self.gpu_no = gpu_no
			self.use_cuda = use_cuda

		if (type(outputHeight) is not int) or (outputHeight <= 0):
			raise Exception("Bad argument: 'outputHeight' should be a positive integer.")
		if (type(outputWidth) is not int) or (outputWidth <= 0):
			raise Exception("Bad argument: 'outputWidth' should be a positive integer.")
		if ((type(outputPixel_dx) is not float) and (type(outputPixel_dx) is not int)) or (outputPixel_dx <= 0):
			raise Exception("Bad argument: 'outputPixel_dx' should be a positive real number.")
		if ((type(outputPixel_dy) is not float) and (type(outputPixel_dy) is not int)) or (outputPixel_dy <= 0):
			raise Exception("Bad argument: 'outputPixel_dy' should be a positive real number.")

		self.outputResolution = (outputHeight, outputWidth)
		self.outputPixel_dx = outputPixel_dx
		self.outputPixel_dy = outputPixel_dy
		self.outputSpacing = (outputPixel_dx, outputPixel_dy)

		outputGridX, outputGridY = generateGrid(self.outputResolution, outputPixel_dx, outputPixel_dy)
		self.outputGridX = outputGridX.to(device=self.device)
		self.outputGridY = outputGridY.to(device=self.device)

		self.calculateOutputCoordGrid()
		
		self.grid = None
		self.prevFieldSpacing = None
		self.prevFieldSize = None


	def calculateOutputCoordGrid(self):
		# Can assume that coordinate (0,0) is in the center due to how generateGrid(...) works
		gridX = self.outputGridX
		gridY = self.outputGridY

		grid = torch.zeros(self.outputResolution[0], self.outputResolution[1], 2)

		# Stuff is ordered this way because torch.nn.functiona.grid_sample(...) has x as the coordinate in the width direction
		# and y as the coordinate in the height dimension.  This is the opposite of the convention used by this code.
		grid[:,:,0] = gridY
		grid[:,:,1] = gridX

		self.gridPrototype = grid.to(device=self.device)
	
	
	def forward(self, field):
		# convert field to 4D tensor for batch processing
		Bf,Tf,Pf,Cf,Hf,Wf = field.data.shape
		field_data = field.data.view(Bf*Tf*Pf,Cf,Hf,Wf) # Shape to 4D

		# convert spacing to 4D tensor
		spacing_data = field.spacing.data_tensor.view(field.spacing.tensor_dimension.get_new_shape(new_dim=Dimension.BTPCHW))
		Bs,Ts,Ps,Cs,Hs,Ws = spacing_data.shape
		spacing_data = spacing_data.view(Bs*Ts*Ps,Cs,Hs,Ws)

		buildGridFlag = False
		if (self.grid is None):
			# No grid was ever made, so must make one
			buildGridFlag = True
		elif (self.prevFieldSpacing is None): # This 'elif' is redundant as (self.prevFieldSpacing is None) if and only if (self.grid is None)
			buildGridFlag = True
		elif (self.prevFieldSize is None): # This 'elif' is redundant as (self.prevFieldSize is None) if and only if (self.grid is None)
			buildGridFlag = True
		elif not (torch.equal(self.prevFieldSpacing, field.spacing.data_tensor)):
			buildGridFlag = True
		elif not (torch.equal(torch.tensor(self.prevFieldSize), torch.tensor(field.data.shape))):
			buildGridFlag = True

		if (buildGridFlag):
			# Calculating stuff for normalizing the output coordinates to the input coordinates
			xNorm = spacing_data[:,:,0,:] * ((Hf - 1) // 2)
			xNorm = xNorm[:,:,None,:]
			yNorm = spacing_data[:,:,1,:] * ((Wf - 1) // 2)
			yNorm = yNorm[:,:,None,:]

			self.grid = self.gridPrototype.repeat(Bf*Tf*Pf,1,1,1)

			# Stuff is ordered this way because torch.nn.functiona.grid_sample(...) has x as the coordinate in the width direction
			# and y as the coordinate in the height dimension.  This is the opposite of the convention used by this code.
			self.grid[... , 0] = self.grid[... , 0] / yNorm
			self.grid[... , 1] = self.grid[... , 1] / xNorm

		self.prevFieldSpacing = field.spacing.data_tensor
		self.prevFieldSize = field.data.shape
		
		new_data = grid_sample(field_data.real, self.grid, mode='bilinear', padding_mode='zeros', align_corners=True)
		new_data = new_data + (1j * grid_sample(field_data.imag, self.grid, mode='bilinear', padding_mode='zeros', align_corners=True))

		# This is less efficient with GPU memory:
			# new_data_real = grid_sample(field_data.real, self.grid, mode='bilinear', padding_mode='zeros', align_corners=False)
			# new_data_imag = grid_sample(field_data.imag, self.grid, mode='bilinear', padding_mode='zeros', align_corners=False)
			# new_data = new_data_real + (1j * new_data_imag)

		new_data = new_data.view(Bf,Tf,Pf,Cf,self.outputResolution[0],self.outputResolution[1]) # Reshape to 6D

		# Assumes that the last dimension of the input field's spacing data tensor contains the x- and y-spacings
		new_spacing_data = torch.clone(field.spacing.data_tensor)	# Apparently, before clone() was used, new_spacing_data shared a pointer with field.spacing.data_tensor
																	# This caused unexpected behavior.
		new_spacing_data[... , 0] = self.outputPixel_dx
		new_spacing_data[... , 1] = self.outputPixel_dy

		spacing = SpacingContainer(spacing=new_spacing_data, tensor_dimension=field.spacing.tensor_dimension)
		spacing.set_spacing_center_wavelengths(spacing.data_tensor)

		Eout = 	ElectricField(
					data = new_data,
					wavelengths = field.wavelengths,
					spacing = spacing
				)

		return Eout