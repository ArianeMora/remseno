# Import classifiers
from remseno import *
import numpy as np


np.seterr(divide='ignore', invalid='ignore')

# The file with the coordinates for the trees in it
coordinate_file_path = 'data/Allen_all_genotyped_trees_2024.csv'

# just add in all the files you want to add here!
training_times = ['data/20230421_Allen.tif',
                  'data/20230708_Allen.tif',
                  'data/20231002_Allen.tif']


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
coords = []
for data_file in training_times:
    o = Image()
    o.load_image(image_path=data_file)
    # Get the indicies from the image
    indices = get_all_planetscope(o.image)
    bands = [o.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]] + list(indices)
    ndvi = get_ndvi(o.image, red_band=6, nir_band=8)
    mask = ndvi > 1 # Change this!!
    bands_dict = {}
    for i, b in enumerate(bands):
        bands_dict[i] = b*mask
    images.append({'image': o, 'indexs': bands_dict})
    # Also plot the image as rbg + the mask!
    o.plot_rbg(show_plot=True)

# Get the coordinates of all pixels with the mask values > than the amount!
coords = set(coords) # Now write though we don't know what they are, we'll just classify them all as a single type

c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
           id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:32632")

c.transform_coords(tree_coords="EPSG:32632", image_coords="EPSG:32632", plot=False)

train_df = build_dataset_across_images(c.df, images, c, max_pixel_padding=1, normalise=True)

# Run the k-fold cross validation.
ml = ML()
classifiers = ml.train_clf(train_df, 'Sylvatica', 'Orientals',
                           csv_file="classifier_metrics_test_set_masked.csv")

# Make the validation dataset using other images
testing_times = ['data/20230518_Allen.tif',
                 'data/20230816_Allen.tif',
                 'data/20231013_Allen.tif']

images = []
for data_file in testing_times:
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

validation_df = build_dataset_across_images(c.df, images, c, max_pixel_padding=1, normalise=True)

ml.validate_clf(validation_df, classifiers, class1_label='Sylvatica', csv_file="classifier_metrics_validation_set.csv")

