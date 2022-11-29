##########
#### This Package contains components to build optical setups. 
#### Absorption grating class
####Author: Lionel Fiske 
####Last update 9/15/2021
#####


import torch
import numpy as np
import math    
import warnings
from Helper_Functions import * 

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 


class Absorption_Mask(torch.nn.Module):
    
    def __init__(self, transmission ,  device = torch.device("cpu") ,  fixed_pattern = False, dtype=torch.double):
        """
        Applies ideal absorption mask to the incident field. 
        
        
        transmission       : float Tensor
                           transmission coefficients to be multiplied in. Internally clamped to between 0-1.
        
        
        device             :  torch device
        
        fixed_pattern      : Bool
                           If True the phase delay will not be set to an nn.parameter to be optimized for 
                   
                   
        """
        super().__init__()
        
        
        if fixed_pattern== False:
            self.transmission = torch.nn.Parameter(transmission.to(device),requires_grad=True).type(dtype)
        else:
            self.transmission = transmission.to(device).type(dtype)
          
          
        #Set internal variables
        self.device = device
        self.fixed_pattern = fixed_pattern
        
        
        
    def forward(self, field):
        """
        Takes in complex tensor and applies an absorption mask to it. 
        
        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).
        """
        Eout_trans =   field*self.transmission.clamp(0,1)
        
        if field.ndim == 2:
            return Eout_trans.squeeze()
        else:
            return Eout_trans 

        return Eout_trans 

