# LocScale-SURFER 
## ChimeraX Bundle 

<img src="./docs/img/locscale_surfer.png" width="250" align="right">

`LocScale-SURFER` is a ChimeraX bundle for enhancing visualisation and interpretation of LocScale2.0-optimised maps containing contextual structure.  

It currently supports the detection and targeted segmentation/removal of micellar/membranous structures, but may be extended to other contextual structures in the future. 

`LocScale-SURFER` runs on Linux, MacOS or Windows WSL platforms and is GPU-accelerated if GPU is available, but also runs on CPUs if required.

## Documentation

>[!IMPORTANT]
> Please visit [https://cryotud.github.io/locscale-surfer/](https://cryotud.github.io/locscale/) for comprehensive documentation, tutorials and troubleshooting.

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

