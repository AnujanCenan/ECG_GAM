# ECG_GAM


## Necessary Requirements

**.env file**
With the following keys
- MIT_BIH_DATASET
- CHAPMAN_DATASET
- PTBXL_DATASET: path to the ptbxl directory
- MODELS_DIR: path to the directory containing the .h5 files


**models directory**
Containing the .h5 files produced in src/train.py
This is the directory referenced by MODELS_DIR in .env

**Python version**
python 3.10 or 3.11

## Third Party Dependencies
- https://github.com/Aura-healthcare/Fast-QRS-detector