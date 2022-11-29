##########
#### This Package contains components to build optical setups. 
#### This class models gaussian blurring of real and imaginary components separately
#### Author: Ollie Cossairt
#### 8/2021
#####


import torch
import numpy as np
import math    
import warnings
import torchvision

from Helper_Functions import * 

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
    
    
class Gaussian_Blur(torch.nn.Module):
    
    def __init__(self, sigma=2  , device = torch.device("cpu")  ):
        """

        """
        super(Gaussian_Blur, self).__init__()

        self.device = device
        self.sigma = sigma
        self.kw = int(sigma//2)*2 + 1
        
    def forward(self, field):
        """
This function 
        
        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).


        Sigma            : float
                           Standard deviation 
                        
        device           : torch.device                
                       
        """

        
        

        if field.ndim == 2:

            B=torch.squeeze( torchvision.transforms.functional.gaussian_blur(field[None,None,:,:].real, sigma=self.sigma, kernel_size=self.kw))
            Bi=torch.squeeze(torchvision.transforms.functional.gaussian_blur(field[None,None,:,:].imag, sigma=self.sigma, kernel_size=self.kw))
            Eout =   B + 1j*Bi


            return Eout.squeeze()
        else:
            B=torch.squeeze( torchvision.transforms.functional.gaussian_blur(field.real, sigma=self.sigma, kernel_size=self.kw))
            Bi=torch.squeeze(torchvision.transforms.functional.gaussian_blur(field.imag, sigma=self.sigma, kernel_size=self.kw))

            Eout =   B + 1j*Bi
            return Eout 

