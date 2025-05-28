# Installation

### From the ChimeraX toolshed

Navigate to the [ChimeraX toolshed](https://cxtoolshed.rbvi.ucsf.edu/) (menu "Tools" -> "More tools") and install `locscalesurfer`. Alternatively, in the ChimeraX command line interface:

```bash
toolshed install locscale surfer
```

### From source code: 
1. Clone the repository: 
```bash
git clone https://github.com/cryoTUD/locscale-surfer.git
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

