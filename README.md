![Electric Field Bunny](/ElectricFieldBunny.gif?raw=true "Electric Field Bunny")

# Overview

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This repository contains some work I did pertaining to synthetic wavelength holography (SWH) and phase retrieval.  The initial objective of this work was to use [Holotorch](https://github.com/facebookresearch/holotorch) [1]—*a Fourier optics/coherent imaging library*—to implement a subset[^1] of the processing described in [2] and [3].  More specifically, the model-based phase retrieval process described in [2] and [3] was implemented.  Additionally, synthetic wavelength holography from [2] was also implemented.  **_The code can be found in the folder ```CODE AND DATA/SWH Code/```._**  Please be aware though that that the code/comments/documentation for the code is, in some places, rather unpolished.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Currently, the processing code is set up to take synthetic data as its input and output a depthmap (see [2]).  Code for generating synthetic data is also in this repository.  The processing code can be extended to real/experimental data, but code would need to be written to put real data into a format that the processing code can understand—this should be very much doable.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**This repository also contains PowerPoint slides that explain the code in more depth, as well as explain the WISH and WISHED processes detailed in [3] and [2] respectively.**  The slides in ```SLIDES/SWH Code Presentation/SWH Code Presentation.pptx``` give details about how the synthetic data was generated, the models used, and the processing.  The presentation in ```SLIDES/WISHED Presentation/CS496 Presentation.pptx``` provides a lot of background information.  It should be useful for learning about the WISH [3] and WISHED [2] papers, as well as some relevant topics surrounding them—**_in other words, if you are unfamiliar with these topics, the ```CS496 Presentation.pptx``` presentation should be a good place to start._**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As a closing remark, code for generating various figures from the presentations, as well as figures themselves, can be found in the ```SLIDES/``` folder.  **_Please feel free to copy and modify the various MATLAB (\*.m) files in the ```SLIDES/``` folder._**[^3]

&nbsp;

# SWH Code
The SWH code (found in ```CODE AND DATA/SWH Code/```) is broken up into a handful of Python scripts.  The scripts should be run sequentially as the output of one script will be used as the input for the next.

#### The scripts should be run in this order:
1. **```createSyntheticData.py```** — Generates synthetic data
   - **Output:** SLM phase patterns & corresponding sensor intensity images, information about SLM-to-sensor model
   - **Note(s):**
     - The model for the entire system (not just SLM-to-sensor) is specified here
     - There is no analogous script for real, measured data in this repository.
2. **```doPhaseRecovery.py```** — Recovers complex fields at the sensor plane
   - **Input:** SLM phase patterns & corresponding sensor intensity images
   - **Output:** Complex fields at sensor plane (and also complex fields at SLM input)
3. **```analyzeRecoveredFields.py```** — Recovers a depthmap from complex fields at sensor plane
   - **Input:** Complex fields at sensor plane
   - **Output:** Depthmap
   - **Note(s):**
     - Can recover depths for both normal and synthetic wavelengths
4. **```createDepthMap3DModel.py```** — Does further processing on depths, outputs a 3D model
   - **Input:** Depthmap
   - **Output:** A 3D model (\*.stl file) corresponding to the recovered depthmap
   - **Note(s):**
     - Median filtering is performed to remove spikes in depth data
     - Then, Gaussian kernel (filter) is applied (see Section 4.2 in [2])

#### Other Notes
1. Some of the aforementioned scripts have code for plotting in them (some commented, some uncommented)
2. ```plotDataBlahAsdf.py``` contains additional plotting code

&nbsp;

# Fourier Optics/Coherent Imaging Libraries Utilized
### [Holotorch](https://github.com/facebookresearch/holotorch) [1]
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The SWH code in this repository relies heavily on Holotorch.  Holotorch was used to create models for various optical systems.


### [Tocohpy](https://github.com/lfiske1/Tocohpy/tree/master) [4]
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tocohpy was the precursor to Holotorch, and some code from Tocohpy was carried over to Holotorch.  Before I obtained access to Holotorch, I implemented phase retrieval using Tocohpy models.  The code for that can be found in the folder ```SWH/OTHER/Tocohpy Test/```.

&nbsp;

# Holotorch Patches/Extensions
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Although Holotorch was (and still is) extremely useful to me, I ran into a handful of bugs while using it.  When trying to work around these bugs, I specifically avoided modifying any of Holotorch's source code.  Instead, I defined new classes that inherited from the buggy classes and overrode any problematic methods.  These "patched" classes can be found in the folder ```CODE AND DATA/SWH Code/holotorch_addons```.  Moreover, I documented the bugs that I found in the file ```CODE AND DATA/SWH Code/Notes.txt```.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In addition to bug fixes, I also implemented some new classes/components in the ```CODE AND DATA/SWH Code/holotorch_addons``` folder.  Among the most noteworthy are the field resampler, Fresnel two-step propagator, and a thin lens model.

&nbsp;

# Extra Stuff
- A Blender file for generating input images for the synthetic data generator code can be found at ```SWH/OTHER/Depth Map Stuff/Blender/Input_Image_Creator.blend```
  - More information about these input images can be found in 

&nbsp;

# Miscellaneous Notes
- Some data is missing from the ```CODE AND DATA/DATA/``` and ```CODE AND DATA/RESULTS``` folders.  This is because certain files were large and would have taken up too much storage on Github.
  - If you would like some of the missing data, please contact me via email.  My email is listed near the end of this README.

&nbsp;

# Contact Information
**Email:** ```MichaelLee2023@u.northwestern.edu```

&nbsp;

# References

1.	Facebookresearch, “Facebookresearch/holotorch: Holotorch is an optimization framework for differentiable wave-propagation written in PyTorch,” GitHub. [Online]. Available: <https://github.com/facebookresearch/holotorch>. [Accessed: 21-Nov-2022].

2.	Y. Wu, F. Li, F. Willomitzer, A. Veeraraghavan, and O. Cossairt, “WISHED: Wavefront imaging sensor with high resolution and depth ranging,” 2020 IEEE International Conference on Computational Photography (ICCP), 2020.

3.	Y. Wu, M. K. Sharma, and A. Veeraraghavan, “WISH: Wavefront imaging sensor with high resolution,” Light: Science & Applications, vol. 8, no. 1, 2019.

4. L. Fiske, “LFISKE1/Tocohpy at master,” GitHub. [Online]. Available: <https://github.com/lfiske1/Tocohpy/tree/master>. [Accessed: 29-Nov-2022].

5. J. C. Maxwell, “On Physical Lines of Force,” The Scientific Papers of James Clerk Maxwell, 1861. 

&nbsp;

### Footnotes
[^1]: **Stuff from [2] and [3] that is not implemented/utilized by the code in this repository:**
      - Phase unwrapping
      - That procedure described in Section 3.3 of [2] that involves using a larger synthetic wavelength to unwrap the phase from a smaller synthetic wavelength[^2]
      - The procedure described in Section 4.1 of [2] where the complex field at the object plane is recovered using optical system operators and the complex field at the image/sensor plane[^2]
      - Surface roughness simulations, like in [2]
      - **_See ```SLIDES/SWH Code Presentation/SWH Code Presentation.pptx``` for more information_**

[^2]: The code derives depths from phases at the sensor plane (as opposed to recovering the object plane fields like in [2]).

[^3]: For the various MATLAB (\*.m) files in the ```SLIDES/``` folder, I do not particularly care about attribution.  So feel free to use one of those MATLAB files to create your own figure, and feel free to not give attribution.
