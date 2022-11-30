# Overview

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This repository contains some work I did pertaining to synthetic wavelength holography (SWH) and phase retrieval.  The initial objective of this work was to use [Holotorch](https://github.com/facebookresearch/holotorch) [1]—*a Fourier optics/coherent imaging library*—to implement a subset[^1] of the processing described in [2] and [3].  More specifically, the model-based phase retrieval process described in [2] and [3] was implemented.  Additionally, synthetic wavelength holography from [2] was also implemented.  **The code can be found in the folder ```CODE AND DATA/SWH Code/```.**  Please be aware though that that the code/comments/documentation for the code is, in some places, rather unpolished.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Currently, the processing code is set up to take synthetic data as its input and output a depthmap (see [2]).  Code for generating synthetic data is also in this repository.  The processing code can be extended to real/experimental data, but code would need to be written to put real data into a format that the processing code can understand—this should be very much doable.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**This repository also contains PowerPoint slides that explain the code in more depth, as well as explain the WISH and WISHED processes detailed in [3] and [2] respectively.**  The slides in ```SLIDES/SWH Code Presentation/SWH Code Presentation.pptx``` give details about how the synthetic data was generated, the models used, and the processing.  The presentation in ```SLIDES/WISHED Presentation/CS496 Presentation.pptx``` provides a lot of background information.  It should be useful for learning about the WISH [3] and WISHED [2] papers, as well as some relevant topics surrounding them—**in other words, if you are unfamiliar with these topics, the ```CS496 Presentation.pptx``` presentation should be a good place to start.**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As a closing remark, code for generating various figures from the presentations, as well as figures themselves, can be found in the ```SLIDES/``` folder.  **Please feel free to copy and modify the various MATLAB (\*.m) files in the ```SLIDES/``` folder.**[^3]





# Contact
```MichaelLee2023@u.northwestern.edu```



# Sources

1.	Facebookresearch, “Facebookresearch/holotorch: Holotorch is an optimization framework for differentiable wave-propagation written in PyTorch,” GitHub. [Online]. Available: <https://github.com/facebookresearch/holotorch>. [Accessed: 21-Nov-2022].

2.	Y. Wu, F. Li, F. Willomitzer, A. Veeraraghavan, and O. Cossairt, “WISHED: Wavefront imaging sensor with high resolution and depth ranging,” 2020 IEEE International Conference on Computational Photography (ICCP), 2020.

3.	Y. Wu, M. K. Sharma, and A. Veeraraghavan, “WISH: Wavefront imaging sensor with high resolution,” Light: Science & Applications, vol. 8, no. 1, 2019.



&nbsp;

[^1]: **Stuff from [2] and [3] that is not implemented/utilized by the code in this repository:**
      - Phase unwrapping
      - That procedure described in Section 3.3 of [2] that involves using a larger synthetic wavelength to unwrap the phase from a smaller synthetic wavelength[^2]
      - The procedure described in Section 4.1 of [2] where the complex field at the object plane is recovered using optical system operators and the complex field at the image/sensor plane[^2]
      - Surface roughness simulations, like in [2]
      - **See ```SLIDES/SWH Code Presentation/SWH Code Presentation.pptx``` for more information**

[^2]: The code derives depths from phases at the sensor plane (as opposed to recovering the object plane fields like in [2]).

[^3]: For the various MATLAB (\*.m) files in the ```SLIDES/``` folder, I do not particularly care about attribution.  So feel free to use one of those MATLAB files to create your own figure, and feel free to not give attribution.
