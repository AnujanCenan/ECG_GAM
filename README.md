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
python3.11

Can use python3.10 but the libraries in the requirements.txt file will need different (slightly older) versions.

## Third Party Dependencies
- https://github.com/Aura-healthcare/Fast-QRS-detector
- Everything seen in requirements.txt

This project is run as a module
So any python script should be run with ECG_GAM as the current working directory. 

Any python script should be run using the -m option; i.e.
```
python3 -m <path to script>
```

For example, to run train.py you should run
```
python3 -m src.train
```

*Note the use of . to indicate stepping into a directory and notice the lack of .py at the end*