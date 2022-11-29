##########
#### This Package contains components to build optical setups. 
#### Interpolates a complex field to new grid
####Author: Lionel Fiske 
####Last update 9/14/2021
#####


import torch
import numpy as np
import math    
import warnings
from Helper_Functions import * 

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

    
    
class Resize_Field(torch.nn.Module):
    
    def __init__(self, scale_factor = None, size = None , recompute_scale_factor = False  , device = torch.device("cpu"), mode="bicubic"  ):
        """
        Resizes a field 
        Parameters
        ==========
        scale_factor   : float
                        resizing scale factor - optional with size (see torch documentation for interpolate)
                           
        size           :   float
                        new field size - optional with scale factor
                          
        recompute_scale_factor  : bool                  
                          
                       

        """
        super().__init__()

        self.device = device
        self.scale_factor = scale_factor
        self.recompute_scale_factor = recompute_scale_factor
        self.size = size
        self.mode = mode
    def forward(self, field):
        """
        In this function we interpolate a complex field to a new grid
        
        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).

                       
        """

        
        if field.ndim == 2:

            B=torch.squeeze( torch.nn.functional.interpolate(field[None,None,:,:].real, scale_factor =self.scale_factor,  size =self.size, mode=self.mode) )
            Bi=torch.squeeze(torch.nn.functional.interpolate(field[None,None,:,:].imag, scale_factor =self.scale_factor, size =self.size, mode=self.mode) )
            Eout =   B + 1j*Bi


            return Eout.squeeze()
        else:
            B=torch.squeeze( torch.nn.functional.interpolate(field.real, scale_factor =self.scale_factor, size =self.size, mode=self.mode ) )
            Bi=torch.squeeze(torch.nn.functional.interpolate(field.imag, scale_factor =self.scale_factor, size =self.size, mode=self.mode ) )

            Eout =   B + 1j*Bi
            return Eout 
