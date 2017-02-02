# Python tools for aligning Scanning Precession Electron Diffraction (SPED) stacks

##Contents
* Short Description
* Main Tools
* Tasks
* Issues


## Short Description
In some cases, either due to misalignments, sample thickness or geometry, the diffraction pattern in each scan pixel acquired during a SPED experiment in a Transmission Electron Microscope (TEM) may shift relative to the other patterns. This repository contains python tools to fit a two dimensional gaussian to a subset of each diffraction pattern in order to find these relative shifts.

See the <a href="https://github.com/EmilChristiansenNTNU/align_sped/Doc/blob/README-revamp/DOCUMENTATION.md">Documentation</a> for details on each tool.

## Main Tools
* `align_sped.py`
    * Main tools
      * `gaussian_2d()`
      * `fit_gaussian_2d_to_imagesubset()`
* `main.py`
    * Makes MxN test data (image)
    * Fits a gaussian to the test data
    * Plots the result
    
### Minor Tools
* Testing
  * `test_gaussian_2d.py` for unit testing input and output of 2D gaussian function

## Tasks
- [ ] Finish fitting function
- [ ] Implement more and better unit testing gases and functions
- [ ] Provide documentation on all tools

## Issues
