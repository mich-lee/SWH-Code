import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

import warnings

import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
from holotorch.utils.Enumerators import *
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.Light import Light

from holotorch_addons.Field_Resampler import Field_Resampler

########################################################################################################################

class Multi_Resolution_Sequential(CGH_Component):
	"""
	Similar to torch.nn.Sequential, but this allows for rescaling/resampling the fields in-between steps.
	Note that this 
	"""
	def __init__(self,
					components				: list or tuple,
					resolutions			: list or tuple,
					elementSpacings		: list or tuple,
					device				: torch.device = None,
					gpu_no				: int = 0,
					use_cuda			: bool = False
				) -> None:

		def validateComponentsList():
			for componentElement in components:
				if not isinstance(componentElement, CGH_Component):
					return False
			return True

		def validateListOf2TupleList(l):
			for elem in l:
				# Not bothering to check if elements of elem are numbers.
				if (type(elem) is not list) and (type(elem) is not tuple):
					return False
				if (len(elem) != 0) and (len(elem) != 2):
					return False
			return True

		def validateArguments():
			if ((type(components) is not list) and (type(components) is not tuple)) or (validateComponentsList() == False):
				raise Exception("'components' argument must be a list or tuple containing objects of type torch.nn.Component (or a subclass of it).")
			if (len(components) + 1) != len(resolutions):
				raise Exception("'resolutions' argument must be a list of length len(components)+1.")
			if (not validateListOf2TupleList(resolutions)):
				raise Exception("'resolutions' argument must be a list of lists, where each list in the list is either empty or contains two positive integers.")
			if (len(components) + 1) != len(elementSpacings):
				raise Exception("'elementSpacings' argument must be a list of length len(components)+1.")
			if (not validateListOf2TupleList(elementSpacings)):
				raise Exception("'elementSpacings' argument must be a list of lists, where each list in the list is either empty or contains two positive real numbers.")
			for i in range(len(resolutions)):
				if (len(resolutions[i]) != len(elementSpacings[i])):
					raise Exception("Mismatched specifications for 'resolutions' and 'elementSpacings'.  Elements at each corresponding position of those lists must either be empty or contain two elements.")

		if (device != None):
			self.device = device
		else:
			self.device = torch.device("cuda:"+str(gpu_no) if use_cuda else "cpu")
			self.gpu_no = gpu_no
			self.use_cuda = use_cuda

		validateArguments()
		super().__init__()

		self.model = torch.nn.Sequential()
		for i in range(len(components)):
			if len(resolutions[i]) != 0:
				tempResampler =	Field_Resampler(
									outputHeight = resolutions[i][0],
									outputWidth = resolutions[i][1],
									outputPixel_dx = elementSpacings[i][0],
									outputPixel_dy = elementSpacings[i][1],
									device = self.device
								)
				self.model.append(tempResampler)
			self.model.append(components[i])
		
		if len(resolutions[-1]) != 0:
			tempResampler =	Field_Resampler(
								outputHeight = resolutions[-1][0],
								outputWidth = resolutions[-1][1],
								outputPixel_dx = elementSpacings[-1][0],
								outputPixel_dy = elementSpacings[-1][1],
								device = self.device
							)
			self.model.append(tempResampler)
			

	def forward(self, field : Light) -> Light:
			return self.model(field)