# LocScale-SURFER<br><sup>**S**egmentation of **U**nresolved **R**egions and **F**iltering for **E**nhanced **R**epresentation</sup>

LocScale-SURFER is a [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) bundle for enhancing representation of transmembrane regions of membrane proteins. It is trained to segment voxels corresponding to the micelle belt of an unsharpened cryo-EM reconstruction. The segmented map can then be used to remove micelle densities from the target map. 

![Overview of LocScale-SURFER](./src/images/figure_1-01.png)

Note, there are two ways to speed up the computation. One is to provide a mask which restricts the prediction of detergent micelle to the region of interest. The other is to use a GPU for computation. By default, the tool uses a GPU if available. 

<div style="display: flex; flex-direction: column; align-items: left;">
  <div class="c-compare" style="--value:50%; position: relative; width: 400px; height: 333px; overflow: hidden;">
    <img class="c-compare__left"
         src="img/emd19995_SURFER_pre.png"
         alt="Raw map"
         style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;" />

    <img class="c-compare__right"
         src="img/emd19995_SURFER_post.png"
         alt="Feature-enhanced map"
         style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; clip-path: inset(0 0 0 var(--value));" />

    <input type="range" class="c-compare__range" min="0" max="100" value="50"
           oninput="this.parentNode.style.setProperty('--value', this.value + '%')"
           style="position: absolute; bottom: 10px; left: 10px; width: 90%; z-index: 10;" />
  </div>
</div>
