# Python tools for aligning Scanning Precession Electron Diffraction (SPED) stacks

## Contents
* Disclaimer
* Short Description
* Main Tools
* Tasks
* Issues

## Disclaimer
The code has not been significantly tested and its robustness cannot be guaranteed. When using this software with e.g.
scientific data, be sure to backup your raw data first, in the case this software tampers with things it really should not.
I am not responsible for any eventual artefacts introduced by the code on scientific data, or the loss or corruption of eventual data
and/or metadata.

## Short Description
In some cases, either due to misalignments, sample thickness or geometry, the diffraction pattern in each scan pixel acquired during a SPED experiment in a Transmission Electron Microscope (TEM) may shift relative to the other patterns. This repository contains python tools to fit a two dimensional gaussian to a subset of each diffraction pattern in order to find these relative shifts.

See the <a href="https://github.com/emichr/align_sped/blob/master/Doc/align_sped.pdf">Documentation</a> for details on each tool.

## Tasks
- [ ] Provide documentation on all tools
- [ ] Fix Readme.md (reflect the documentation.pdf version)
- [ ] Further optimize methods

## Issues
