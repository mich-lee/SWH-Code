##########
#### This Package contains components to build optical setups. 
#### Thin lens class applies a quadratic phase delay to the field.
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

    
    
class Thin_Lens(torch.nn.Module):
    
    def __init__(self, f, wavelength, R2, device = torch.device("cpu"), dtype=torch.complex128):
        """
    Applies a quadratic phase to the field from Goodman's Introduction to Fourier Optics Ch5 pg 99         
        Parameters
        ==========
        f              : float
                        Focal length of lens (matching units with wavelength)
                           
        wavelength     :   float
                        wavelength
                          
        R2             : float tensor                  
                       Aperture coordinates in radius    
                       

        """
#         super(Thin_Lens, self).__init__()
        super().__init__()


        self.device = device
        self.phase_delay = (torch.exp( -np.pi*1j*R2 / wavelength / f ) ).to(device)
        
        
    def forward(self, field):
        """
        In this function we apply a phase delay which depends on the local intensity of the electric field at that point
        
        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).

                       
        """
        Eout =   1*field*self.phase_delay

        if field.ndim == 2:
            return Eout.squeeze()
        else:
            return Eout 
        
       
    