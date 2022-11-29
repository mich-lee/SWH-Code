import numpy as np
import sys
import torch
from torch.nn.functional import grid_sample
import matplotlib.pyplot as plt

sys.path.append("holotorch-lib/")
sys.path.append("holotorch-lib/holotorch")

from holotorch.Spectra.SpacingContainer import SpacingContainer

from holotorch.Optical_Components.CGH_Component import CGH_Component
import holotorch.utils.Dimensions as Dimension
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Sensors.IntensityOperator import IntensityOperator
from holotorch.CGH_Datatypes.IntensityField import IntensityField


class SimpleDetector(CGH_Component):
	def __init__(self, outputDatatype = None) -> None:
		super().__init__()
		self.intensity_operator = IntensityOperator() # NOTE: This does NOT normalize by pixel size.
		# self.outputDatatype = outputDatatype
	
	
	def forward(self, field : ElectricField) -> IntensityField:
		fieldOut = self.intensity_operator(field)
		# if (self.outputDatatype is not None):
		# 	fieldOut.data = fieldOut.data.to(self.outputDatatype)
		return fieldOut