# RobustBench

Usage: 

```
./setup.sh
for f in $(ls slurm_submit*sh); do ./$f; done      # Locally
for f in $(ls slurm_submit*sh); do sbatch $f; done # On a Slurm cluster
```

The script `predict.py` obtains model predictions (note this requires CUDA) and saves them to CSV files along with either the input datapoints, or embedding vectors of the input datapoints obtained with [DINOv2](https://dinov2.metademolab.com/). 

`predict.py` is called by the provided `slurm_submit_*.sh` shellscripts. These scripts should also work outside a Slurm environment. Note that the corruptions data in particular may take some time to obtain, as each corruption type and severity is its own dataset.

The script `cut_dims.py` was used to create lower-dimensional versions of the ImageNet datasets. This data was used to determine Clemont's scaling in dimensions, corresponding to the rightmost plots of Figure 5 in our paper.

