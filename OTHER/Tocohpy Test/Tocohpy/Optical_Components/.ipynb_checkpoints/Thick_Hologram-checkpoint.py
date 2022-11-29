##########
#### This Package contains components to build optical setups. 
#### SLM / Phase grating class. This component applies a learnable pointwise phase delay to the field
#### NOTE: THIS CLASS IS IN BETA AND NOT VERIFIED
####Author: Lionel Fiske 
####Last update 9/15/2021
#####



import torch
import numpy as np
import math    
import warnings
from Optical_Components import SLM
from Helper_Functions import * 
from Optical_Propagators import ASM_Prop
from collections import OrderedDict


warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 



class Thick_Hologram(torch.nn.Module):
    
    def __init__(self, phase_delay, dx, dz , wavelength  , H=None, padding = 1/2, device = torch.device("cpu"), dtype=torch.complex128, fixed_pattern = False ):
    
   
        """
        Applies phase delay to the incident field and propagates through to simulate a thick hologram 
        Using a beam propagation method.
        Parameters
        ==========
        phase_delay        : float Tensor Npix , Nypix, Nsteps
                            Phase delay tensor applied 

                           
        device             :  torch device
        
                wavelength        : float
                           Wavelength of light
        
        dx                : float
                           Pixel Size (same units as wavelength)
                           
        dz                : float
                           depth resolution over which hologram is constant (same units as wavelength)
                           
        distance          : float
                          Propagation Distance
       
        N                  : int 
                          Simulation size
        
                                     
        H                 : torch.complex128
                          Simulation size
                         
        padding           : float
                          percent of domain to pad for simulation
                                           
        
        
        fixed_pattern      : Bool
                           If True the phase delay will not be set to an nn.parameter to be optimized for 
                   
        """

        super(Thick_Hologram, self).__init__()
        
        #Set internal variables

        self.device = device
        self.fixed_pattern = fixed_pattern
        self.dtype = dtype
        self.initial_phase_delay = phase_delay
        self.Optical_Path = OrderedDict()    

        # Set up a differentiable BPM data graph 
        for i in range(phase_delay.shape[2]):
            self.Optical_Path['S0-i'] =SLM( phase_delay = phase_delay[:,:,i] ,  device =device )
            self.Optical_Path['P0-i'] =ASM_Prop( wavelength = wavelength,dx = dx, distance = dz , N=phase_delay.shape[0],dtype = torch.complex128 , H=None, device = device)

        
        self.model = torch.nn.Sequential(self.Optical_Path)
        
        
    def forward(self, field):
        """
        Takes in complex tensor and applies a thick hologram with a learnable pattern
        
        Inputs
        ==========
        field            : torch.complex128
                           Complex field (MxN).
        """
        
        Eout =  self.model(field)
        

        if field.ndim == 2:
            return Eout.squeeze()
        else:
            return Eout

        
