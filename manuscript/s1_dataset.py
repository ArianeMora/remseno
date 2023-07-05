"""
# Dataset generation from the Tallo and Neon datasets dataset

## Neon dataset:
Downloaded on 04/07/2023:
https://github.com/weecology/NeonSpeciesBenchmark/tree/main/data/raw/neon_vst_data_2022.csv

## Tallo database:
Downloaded on 04/07/2023:
https://zenodo.org/record/6637599
https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.16302

"""
import pandas as pd

data_dir = '../data/to_publish/'

# Read in Tallo and Neon
tallo_df = pd.read_csv(f'{data_dir}Tallo.csv')
neon_df = pd.read_csv(f'{data_dir}neon_vst_data_2021.csv')

# Neon uses a different species labelling so we need to update this
tallo_df['scientificName'].value_counts()