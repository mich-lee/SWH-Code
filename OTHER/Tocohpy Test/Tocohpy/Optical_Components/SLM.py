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
from Helper_Functions import * 

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 



class SLM(torch.nn.Module):
    
    def __init__(self, phase_delay ,  device = torch.device("cpu"), dtype=torch.double, fixed_pattern = False ):
        """
        Applies phase delay to the incident field. 
        
        Parameters
        ==========
        phase_delay        : float Tensor
                            Phase delay tensor applied 

                           
        device             :  torch device
        
        
        fixed_pattern      : Bool
                           If True the phase delay will not be set to an nn.parameter to be optimized for 
                   
        """
        
        super().__init__()
        
        
        # Check if parameter
        if fixed_pattern == False:
            self.phase_delay = torch.nn.Parameter(phase_delay.to(device).type(dtype),requires_grad=True)
        else :
            self.phase_delay = phase_delay.to(device).type(dtype)

        
        self.device = device
        self.fixed_pattern = fixed_pattern
        self.dtype = dtype
        
    def forward(self, field):
        """
        Takes in complex tensor and applies a phase delay
        
        Inputs
        ==========
        field            : torch.complex128
                           Complex field (MxN).
        """
        
        Eout_SLM = field*torch.exp(1j * self.phase_delay )


        if field.ndim == 2:
            return Eout_SLM.squeeze()
        else:
            return Eout_SLM 

        
