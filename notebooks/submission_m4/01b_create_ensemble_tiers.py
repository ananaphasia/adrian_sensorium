# %% [markdown]
# # Create ensemble tier file

# %%
import os
if 'notebooks' in os.getcwd():
    os.chdir('../..')  # change to main directory
if 'adrian_sensorium' not in os.getcwd():
    if os.getcwd().split('\\')[-1] == "Petreanu Lab":
        os.chdir('adrian_sensorium')
    else:
        raise FileNotFoundError("The path needs to be fixed")
print('Working directory:', os.getcwd())

# %%
import numpy as np
import matplotlib.pyplot as plt
import glob

# %%
%matplotlib inline

# %%
# folders = sorted( glob.glob( "notebooks/data/static*/") )
# folders

# %%
basepath = "notebooks/data/IM_prezipped"
# Add Add folders two levels deep from basepath into a list
# First level
folders = [os.path.join(basepath, name) for name in os.listdir(
    basepath) if os.path.isdir(os.path.join(basepath, name)) and not "merged_data" in name]
# Second level
folders = [os.path.join(folder, name) for folder in folders for name in os.listdir(
    folder) if os.path.isdir(os.path.join(folder, name)) and not "merged_data" in name]
folders = [x.replace("\\", "/") for x in folders]
folders

# %%
# helper function for plotting
def tier_to_int(tier_str):
    tier_int = np.zeros_like(tier_str, dtype=int)
    tier_int[tier_str == 'train'] = 0
    tier_int[tier_str == 'validation'] = 1
    tier_int[tier_str == 'test'] = 2
    tier_int[tier_str == 'final_test'] = 3
    return tier_int

# %%
for folder in folders:
    print('Working on: ', folder)

    # read in original tiers
    tier_file = '/meta/trials/tiers.npy'
    tier_raw = np.load(os.path.join(folder + tier_file))

    # create 5 shuffles of train/val while keeping test data in place
    ensemble_tier = np.zeros((5, len(tier_raw)), dtype='<U10')
    train_val_locs = (tier_raw == 'train') | (tier_raw == 'validation')
    to_shuffle = tier_raw[train_val_locs]

    # seed here to make it independent of order of folders
    np.random.seed(35382)
    for i in range(5):
        ensemble_tier[i, :] = np.copy(tier_raw)
        ensemble_tier[i, train_val_locs] = np.random.permutation(to_shuffle)

    # save array
    np.save(folder + '/meta/trials/ensemble_tiers.npy', ensemble_tier)

    # double check that test data remains not shuffled
    plt.figure(figsize=(12, 4))
    plt.plot(tier_to_int(tier_raw))
    plt.plot(tier_to_int(ensemble_tier[2, :]))
    plt.plot(tier_to_int(ensemble_tier[3, :]))
    plt.title(folder)
    plt.legend(['train', 'validation', 'test', 'final_test'])
    plt.xlim((0, 1000))

# %%



