import torch
import numpy as np
import math    
import warnings



##########
#### This Paackage contains functions to propagate optical fields 
#### Two Step Fresnel transform with fixed coordinates
####Author: Lionel Fiske
####Last update 9/15/2021 - Lionel Fiske
#####

import torch
import numpy as np
import math    
import warnings
from Helper_Functions import * 

from .One_Step_Prop_NC import *


class Two_Step_Prop(torch.nn.Module):
    
    def __init__(self, wavelength, dx, distance , N , H=None, padding = 1/2 ,device = torch.device("cpu")):
        
        """
        Two step Fresnel Propagator adapted from Numerical Simulation of Optical Systems by Jason Schmidt Chapter 6 page 91 to preserve dx

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
        
        super(Two_Step_Prop, self).__init__()
        
        self.wavelength = wavelength
        self.dx1 = dx
        self.distance = distance
        self.device = device

        d1 = 1/2*distance
        d2 =  distance - d1
        
        self.b1= One_Step_Prop_NC( wavelength= wavelength, dx = dx, distance=d1 , N=N , H=None, padding = padding ,dtype=dtype,device = device)
        self.b2= One_Step_Prop_NC( wavelength= wavelength, dx = self.b1.dx_new, distance=d2 , N=N , H=None, padding = padding,dtype=dtype,device = device)

        
    def forward(self, field):
        """
        Two step method adapted from Numerical Simulation of Optical Wave Propagation Jason D. Schmidt

        
        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).
                       
        """
        
        out = self.b2( self.b1(field) )
        
        
        if field.ndim == 2:
            return out.squeeze()
        else:
            return out 



