# import sys
import torch
import numpy as np
import math    
import warnings

# sys.path.append('../Tocohpy')
# import Optical_Propagators as prop
# from Helper_Functions import *

class inverseObject:
	@staticmethod
	def calculate_inverse_asm_prop(asmProp, N, device = torch.device("cpu")):
		Nx = N
		Ny = N

		field_in = torch.zeros((Nx,Ny), dtype = torch.cfloat, requires_grad = False).to(device)

		prev_ix = 0
		prev_iy = 0
		for ix in range(0,Nx-1):
			for iy in range(0,Ny-1):
				field_in[prev_ix,prev_iy] = 0
				field_in[ix,iy] = 1
				asmProp.forward(field_in)