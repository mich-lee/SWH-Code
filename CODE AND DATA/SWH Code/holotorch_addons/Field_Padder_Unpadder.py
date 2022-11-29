import torch
from torch.nn.functional import pad
import warnings
from holotorch.Optical_Components.CGH_Component import CGH_Component

from holotorch.utils.Dimensions import *
from holotorch.utils.Helper_Functions import *
from holotorch.CGH_Datatypes.ElectricField import ElectricField

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Field_Padder_Unpadder(CGH_Component):

	def __init__(	self,
					pad_x : int,
					pad_y : int,
					pad_x1 : int = None,
					pad_y1 : int = None,
					pad_val = 0
				):
		"""
		Pads a field
		Parameters
		==========
		pad_x		:	int
		pad_y		:	int
		pad_x1		:	int = None
		pad_y1		:	int = None
		"""
		super().__init__()

		self.pad_x = pad_x
		self.pad_y = pad_y
		if (pad_x1 is None):
			self.pad_x1 = pad_x
		else:
			self.pad_x1 = pad_x1
		if (pad_y1 is None):
			self.pad_y1 = pad_y
		else:
			self.pad_y1 = pad_y1
		self.pad_val = pad_val


	def forward(self, field : ElectricField or torch.Tensor) -> ElectricField or torch.Tensor:
		if (type(field) is ElectricField):
			field_data = field.data
		else:
			field_data = field

		dataOut = pad(field_data, (self.pad_y, self.pad_y1, self.pad_x, self.pad_x1), mode='constant', value=self.pad_val)

		if (type(field) is ElectricField):
			out = ElectricField(
				data = dataOut.to(field.data.device),
				wavelengths=field.wavelengths,
				spacing = field.spacing,
			)
		else:
			out = dataOut.to(field.device)

		return out