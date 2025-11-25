## Multiply quantized vortex spectroscopy in a quantum fluid of light
This repository contains the codes used to generate the figures of the article from the accompanying data. The datasets, together with the analysis scripts, are openly available on Zenodo [(link)](https://zenodo.org/records/17699486). To reproduce the figures, run all Python scripts from within the `data/` directory.
   
### Codes:
For Experiments:
- `contrast.py` : tools to extract observables from interferograms data
- `velocity.py` : tools to extract observables from field data
- `polar_projection.py` : tools to map cartesian and polar basis fields
- `dict_utilities.py` : tools to save/load dictionary as/from text
- `data_analysis.py` : process all the data and generate figures

For Numerical Simulation:
- `physical_cst.py` : contains physical constants and sample parameters
- `vortex_simu.py` : run the simulations, process the data and generate figures

### Experiments
The dataset contains experimental data used to generate MainFig1,2,3, and SMFig3,4,5 of the article. Includes raw interferogram images, density images and processed data when convenient.
  - 20250410_serie3 : Charge 4 MQV (MainFig1,2,3, SMFig4)
  - 20250405_serie1 : Empty cavity (polariton mass estimation)
  - 20250405_serie2 : Resting fluid (reservoir estimation, SMFig3)
  - 20250810_serie0 : OAI illustration (SMFig5)

#### Interferogram data:
Each file `interferograms.h5` stores the raw interferograms for a frequency scan of one spatial probe configuration.
  - **Shape:** `(N_shots, Nx, Ny)`:
      - `Nx`: pixel number along x
      - `Ny`: pixel number along y
      - `N_shots`: number of camera shots along the frequency scan 
  - **Content:** raw interferogram images; `interferograms[n]` returns the frame of the n-th frequency value.

#### Camera data:
Each file `density.tiff` contains an optical density taken by the camera.
  - **Shape** `(Nx, Ny)`:
      - `Nx`: pixel number along x
      - `Ny`: pixel number along y

#### Figures generation
Running `data_analysis.py` reproduces all experimental plots and 2D-maps of the article. The figures (data/Combined Figures folder) are then assembled using Inkscape.

### Numerical Simulation
The dataset is fully generated and analyzed running `vortex_simu.py`. Requires a GPU device and installation of the python package [ddGPE](https://github.com/Quantum-Optics-LKB/ddGPE).

#### Figures generation
Running `vortex_simu.py` with python launch from the cloned repository reproduces all numerical plots and 2D-maps of SMFig2. The figure is combined (data/Combined Figures folder) using Inkscape.
