# Import classifiers
from remseno import *
import numpy as np


np.seterr(divide='ignore', invalid='ignore')

# The file with the coordinates for the trees in it
coordinate_file_path = 'data/Allen_all_genotyped_trees_2024.csv'

# just add in all the files you want to add here!
training_times = ['data/20230421_Allen.tif', 'data/20230518_Allen.tif', 'data/20230915_Allen.tif',
                  'data/20230708_Allen.tif', 'data/20230816_Allen.tif', 'data/20230708_Allen.tif']

def get_all_planetscope(img):
    ndvi = get_ndvi(image=img, red_band=6, nir_band=8)
    sr = get_sr(image=img, red_band=6, nir_band=8)
    pri = get_pri(image=img, green_band=4, greeni_band=3)
    tcari = get_tcari(image=img, rededge_band=7, greeni_band=3, red_band=6)
    redge = get_redge(image=img, nir_band=8, green=4, r_edge=7)
    schl = get_schl(image=img, red_band=6, rededge_band=7, nir_band=8)
    schlcar = get_schlcar(image=img, red_band=6, greeni_band=3)
    return ndvi, sr, pri, tcari, redge, schl, schlcar

images = []
for data_file in training_times:
    o = Image()
    o.load_image(image_path=data_file)
    # Get the indicies from the image
    indices = get_all_planetscope(o.image)
    bands = [o.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]] + list(indices)
    bands_dict = {}
    for i, b in enumerate(bands):
        bands_dict[i] = b
    images.append({'image': o, 'indexs': bands_dict})

c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
           id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:32632")

c.transform_coords(tree_coords="EPSG:32632", image_coords="EPSG:32632", plot=False)

train_df = build_dataset_across_images(c.df, images, c, max_pixel_padding=1, normalise=True)

# Run the k-fold cross validation.
ml = ML()
ml.perform_k_fold_cv(train_df, 'Sylvatica', 'Orientals', csv_file="classifier_metrics_with_per_fold_kfold.csv")