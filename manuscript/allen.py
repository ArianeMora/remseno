import pandas as pd

from remseno import *
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


np.seterr(divide='ignore', invalid='ignore')

coordinate_file_path = 'data/Allen_all_genotyped_trees_2024.csv'
c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
                   id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:32632")

c.transform_coords(tree_coords="EPSG:32632", image_coords="EPSG:32632", plot=True)

#training_times = ['data/20230421_Allen.tif', 'data/20230615_Allen.tif', 'data/20230816_Allen.tif']
training_times = ['data/20230421_Allen.tif']#, 'data/20230615_Allen.tif']#, 'data/20230816_Allen.tif', 'data/20230518_Allen.tif', 'data/20230708_Allen.tif']

test_times = ['data/20230518_Allen.tif']#, 'data/20230708_Allen.tif'] #'data/20230915_Allen.tif']

training_data = []


def get_all_planetscope(img):
    ndvi = get_ndvi(image=img, red_band=6, nir_band=8)
    sr = get_sr(image=img, red_band=6, nir_band=8)
    pri = get_pri(image=img, green_band=4, greeni_band=3)
    tcari = get_tcari(image=img, rededge_band=7, greeni_band=3, red_band=6)
    redge = get_redge(image=img, nir_band=8, green=4, r_edge=7)
    schl = get_schl(image=img, red_band=6, rededge_band=7, nir_band=8)
    schlcar = get_schlcar(image=img, red_band=6, greeni_band=3)
    return ndvi, sr, pri, tcari, redge, schl, schlcar

rows = []
images = []
coords = []
for data_file in training_times:
    o = Image()
    o.load_image(image_path=data_file)

    # Get the indicies from the image
    indices = get_all_planetscope(o.image)
    bands = list(indices) #+[o.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]] #
    bands_dict = {}
    for i, b in enumerate(bands):
        bands_dict[i] = b
    images.append({'image': o, 'indexs': bands_dict})
    rows += bands
    c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
               id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:32632")

    c.transform_coords(tree_coords="EPSG:32632", image_coords="EPSG:32632", plot=False)
    coords.append(c)

clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.2,
    max_depth=3, random_state=42
)

#df = pd.DataFrame(rows)
#df.to_csv('test_rows.csv')
ml = ML()
# train_df = ml.train_ml(clf, image=images[0], coords=c, image_bands=rows, validation_percent=30, test_percent=30,
#                        max_pixel_padding=1, normalise=False)


ml.train_ml_on_multiple_images(clf, images, coords=c, validation_percent=5, test_percent=30,
                                max_pixel_padding=1, normalise=True)

ml.train_df.to_csv('train.csv')

test_images = []
coords = []
rows = []
for data_file in test_times:
    o = Image()
    o.load_image(image_path=data_file)

    # Get the indicies from the image
    indices = get_all_planetscope(o.image)
    bands = list(indices) #+[o.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]] #
    bands_dict = {}
    for i, b in enumerate(bands):
        bands_dict[i] = b
    test_images.append({'image': o, 'indexs': bands_dict})
    rows += bands
    c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
               id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:32632")

    c.transform_coords(tree_coords="EPSG:32632", image_coords="EPSG:32632", plot=False)
    coords.append(c)

ml.train_ml_on_multiple_images(ml.clf, test_images, coords=c, validation_percent=50, test_percent=30,
                       max_pixel_padding=1, normalise=True, pretrained=True)
