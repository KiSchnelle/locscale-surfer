![GitHub Release](https://img.shields.io/github/v/release/cryotud/locscale-surfer)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/pypi/l/locscale.svg?color=orange)](https://gitlab.tudelft.nl/aj-lab/locscale/raw/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15488062.svg)](https://doi.org/10.5281/zenodo.15488062)

# LocScale-SURFER 
## ChimeraX Bundle 

<img src="./docs/img/locscale_surfer.png" width="250" align="right">

`LocScale-SURFER` is a ChimeraX bundle for enhancing visualisation and interpretation of LocScale2.0-optimised maps containing contextual structure.  

It currently supports the detection and targeted segmentation/removal of micellar/membranous structures, but may be extended to other contextual structures in the future. 

`LocScale-SURFER` runs on Linux, MacOS or Windows WSL platforms and is GPU-accelerated if GPU is available, but also runs on CPUs if required.

## Documentation

>[!IMPORTANT]
> Please visit [https://cryotud.github.io/locscale-surfer/](https://cryotud.github.io/locscale-surfer/) for comprehensive documentation, tutorials and troubleshooting.

## Installation

### From source code: 
1. Clone the repository: 
```bash
git clone https://gitlab.tudelft.nl/aj-lab/surfer.git
```
Note the full path of the cloned repository.
PATH_TO_REPO = path/to/surfer

2. Open ChimeraX and navigate to the command line interface.

3. Install the bundle by running the following command:
```chimerax
devel install PATH_TO_REPO
devel clean PATH_TO_REPO
```
4. Restart ChimeraX.
5. You should now see the LocScale-SURFER tool in: ```Tools > Volume Data > LocScale-SURFER ```

## Credits
`LoScale-SURFER` is facilitated by a number of open-source projects.

- [`OPM database`](https://opm.phar.umich.edu/ppm_server): Orientations of Proteins in Membranes (OPM) database.
- [`Optuna`](https://github.com/optuna/optuna): Hyperparameter optimisation. [MIT license]
- [`SCUNet`](https://github.com/cszn/SCUNet): Semantic segmentation. [Apache 2.0]


