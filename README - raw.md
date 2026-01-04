# Esophagus Segmentation Tasks

![Esophagus](./figures/fig1.jpg)

## Train Process
The main training file is:
```
train.py
```
Moreover, if you want to use multi-card parallel training, you can adjust the relevant parameters and execute:
```
train.sh
```
All training and saving parameters are in:
```
config.yml
```
This release keeps only the core method: `EAT`.

## DCNv4 (Required)
`EAT` depends on DCNv4. The source code is provided in `DCNv4_op/` and needs to be compiled/installed by users.

Build & install (CUDA required):
```
cd DCNv4_op
bash make.sh
```

If you prefer editable install:
```
cd DCNv4_op
pip install -v -e .
```

Note: `DCNv4_op/setup.py` requires CUDA and will raise an error if CUDA is not available.

## Optional pretrained weights
`config.yml` points to an optional backbone weight file:
```
./pretrained/pvt_v2_b2.pth
```
If the file is missing, the code will still run and will initialize the backbone without loading pretrained weights.

## Verify Process
The main verifing file is:
```
verify.py
```
Moreover, if you want to use multi-card parallel training, you can adjust the relevant parameters and execute:
```
verify.sh
```

## Datasets
### There are the Esophagus Segmentation Tasks:

### CVC-ClinicDB
1、The dataloader file is 
```
src \ CVCLoder.py
```
2、Please rename the Ground Truth folder in the CVC-ClinicDB decompressed data to GroundTruth, that is, remove the spaces. and fill the root path in the following files:
```
config.yml
```

### Other Task
Please add it according to the above format.\
Especially when adding additional dataloader files, do not modify src\CVCLoder.py

## verification
Users can use the following files to visualize segmentation tasks: 
```
visualization.py
```
It is worth noting that if visualization is required, the user needs to place the image to be segmented under this path:
```
visualization/img
```
After the visualization program is executed, the visualization results will be observed under this path:
```
visualization/output
```







