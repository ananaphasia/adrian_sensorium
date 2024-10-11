# %% [markdown]
# # Train models

# %% [markdown]
# For this submission, 5 models were trained with different train/validation splits and different model seeds.
# 
# To speed up training, the models were trained in parallel on the CSCS infrastructure by calling :
# 
# cd adrian_sensorium/scripts
# 
# bash start_jobs.sh jobs_ensemble.txt
# 
# This script starts 5 machines to run the adrian_sensorium/scripts/train_model.py script with the 5 configuration files in the folder adrian_sensorium/saved_models/config_m4_ens*.yaml
# 
# To reproduce this fitting, one can also execute the following code (not tested):

# %%
import os
if 'notebooks' in os.getcwd():
    os.chdir('../..')  # change to main directory
print('Working directory:', os.getcwd())

# %%
os.getcwd()

# %%
!python scripts/train_model_copy.py -m config_m4_ens0
# !python scripts/train_model.py -m config_m4_ens1
# !python scripts/train_model.py -m config_m4_ens2
# !python scripts/train_model.py -m config_m4_ens3
# !python scripts/train_model.py -m config_m4_ens4


