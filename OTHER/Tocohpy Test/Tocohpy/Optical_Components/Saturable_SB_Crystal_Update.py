##########
#### This Package contains components to build optical setups. 
#### The code contains the nonlinearity for a thin slice of an SB nonlinear crystal
#### The nonlinearity here saturates to a final value as intensity approaches infinity
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


    
class Saturable_SB_Crystal_Update(torch.nn.Module):
    
    def __init__(self, gamma, distance, dtype = torch.complex128, device = torch.device("cpu")  ):
        """
        Solution of the nonlinear refractive update step for use in a Fourier Split step method for solving the NLS equation   Ref:   https://en.wikipedia.org/wiki/Split-step_method

        For use with a short distance prop such as NLS_prop. The form of the nonlinearity is saturable. 
        
        Parameters
        ==========
        Gamma              : float
                           Magnitude of the nonlinearity 
                           
        distance          :   float
                           Size of step 
                          

        """
        super(Saturable_SB_Crystal_Update, self).__init__()
        
        #Set internal variables
        self.gamma = gamma
        self.distance = distance
        self.device = device
        self.dtype = dtype
        
        
    def forward(self, field):
        """
        In this function we apply a saturable phase delay which depends on the local intensity of the electric field at that point
        
        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).

                       
        """
        phase_delay = ( field.abs()**2 * self.gamma * self.distance  / (1+ field.abs()**2 ) )
        Eout =   field*torch.exp(- 1j *phase_delay )



        if field.ndim == 2:
            return Eout.squeeze()
        else:
            return Eout 

