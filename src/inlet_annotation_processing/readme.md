# Inlet Annotation Pipeline

A minimal workflow for annotating **two inlets** (left / right) of coronary artery in 3‑D medical volumes and exporting them as a one‑hot volume.  The procedure is split into three stages:

1. **Pre‑annotation** – sample a point cloud from each volume.
2. **Annotation** – place two key‑points with the [3‑D Annotation Tool](https://github.com/zhuyinheng/3d-annotation-tool-imagecas-annotation#).
3. **Post‑annotation** – convert key‑points back into a voxel mask that shares the original header.

> **Note –** run this pipeline only when you need to inspect or redo the annotations; the final one‑hot inlet masks are already provided.

---

## Quick Start

> **Install the 3‑D Annotation Tool first** — run once per machine  
> ```bash
> git clone https://github.com/zhuyinheng/3d-annotation-tool-imagecas-annotation.git
> cd 3d-annotation-tool-imagecas-annotation
> git submodule update --init --recursive
> git lfs install && git lfs pull
> mkdir build && cd build
> cmake .. && make -j$(nproc) && sudo make install
> ```

```bash
# Pre‑annotation – volumes → point clouds (.ply)
python inlet_annotation_processing.py --pre \
       --imagecas_root ./data/imagecas \
       --annotation_root ./data/inlet_annotation

# Annotate every .ply in the GUI (see §4)

# Post‑annotation – JSON → inlet masks (.nii.gz)
python inlet_annotation_processing.py --post \
       --imagecas_root ./data/imagecas \
       --annotation_root ./data/inlet_annotation
```

Expected tree after successful run:

```
data/
├─ imagecas/                # input *.label.nii.gz
└─ inlet_annotation/
   ├─ *.ply                 # point clouds
   ├─ *.json                # annotations
   └─ *.label.nii.gz        # inlet masks (1 = L, 2 = R)
```

---

## Detailed Workflow

### Pre‑annotation

`inlet_annotation_processing.py --pre` performs:

1. **Marching cubes** to get a surface mesh.
2. **Random sampling** (\~5 k vertices) → point cloud.
3. **Normalisation** into the unit cube:
   $\text{pcd} = \frac{\text{voxel}}{256} - 0.5$

**Why this scaling?** The 3‑D Annotation Tool only renders point clouds normalised to the cube *[‑0.5, 0.5]*³, so voxel indices are mapped to that range before annotation.

4. Save as *.ply* next to the volume name.

If your volumes are not 256³ or isotropic, adjust the divisor in `vol_idx_to_pcd_transform()` (see source code).

---

### Annotation

| Key / Mouse  | Action                                     |
| ------------ | ------------------------------------------ |
| **n / p**    | Next / previous case                       |
| **0 / 1**    | Tag left (class 0) / right (class 1) inlet |
| **Ctrl + S** | Save → *same filename*.json                |

**Guideline** – drop exactly one point inside each inlet; 1–2 voxel accuracy is enough.

---

### Post‑annotation

`inlet_annotation_processing.py --post` reads every *.json*, converts points back to voxels
$\text{voxel} = \operatorname{round}\big((\text{pcd}+0.5)\times256\big)$
and writes a 1‑channel NIfTI mask:

* **1** = left inlet, **2** = right inlet.
  The header (spacing, origin, orientation) comes from the source volume via `save_nii(..., ref_nii_path=...)`.
