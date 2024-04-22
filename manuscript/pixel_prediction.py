import pandas as pd
import numpy as np
from remseno import *


coordinate_file_path = 'data/Allen_all_genotyped_trees_2024.csv'
df = pd.read_csv(coordinate_file_path)
print(np.min(df.X.values), np.min(df.Y.values), np.max(df.X.values), np.max(df.Y.values))

rows = []
# Define the bounds of your area [min_x, max_x, min_y, max_y] in meters
min_x, max_x = 379343, 379631 # example bounds in meters for eastings
min_y, max_y = 5388899, 5389246  # example bounds in meters for northings

# Distance between each point in meters
step = 10

# List to hold coordinates
coordinates = []

# Generate coordinates
i = 0
for y in range(min_y, max_y + 1, step):
    for x in range(min_x, max_x + 1, step):
        if i%2 == 0:
            # Made up classification
            rows.append([i, 'Sylvatica', x, y])
        else:
            rows.append([i, 'Orientals', x, y])

        i += 1

# Just make a coordinate file
df = pd.DataFrame(rows)
df.columns = ['id', 'class', 'X', 'Y']

df.to_csv('data/pixel_prediction.csv', index=False)

# Test the file: this is a copy of training

np.seterr(divide='ignore', invalid='ignore')

# The file with the coordinates for the trees in it
coordinate_file_path = 'data/Allen_all_genotyped_trees_2024.csv'

def get_all_planetscope(img):
    ndvi = get_ndvi(image=img, red_band=6, nir_band=8)
    sr = get_sr(image=img, red_band=6, nir_band=8)
    pri = get_pri(image=img, green_band=4, greeni_band=3)
    tcari = get_tcari(image=img, rededge_band=7, greeni_band=3, red_band=6)
    redge = get_redge(image=img, nir_band=8, green=4, r_edge=7)
    schl = get_schl(image=img, red_band=6, rededge_band=7, nir_band=8)
    schlcar = get_schlcar(image=img, red_band=6, greeni_band=3)
    return ndvi, sr, pri, tcari, redge, schl, schlcar


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

# Now we use the new file for classification!!!
coordinate_file_path = 'data/pixel_prediction.csv'

c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
           id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:32632")

c.transform_coords(tree_coords="EPSG:32632", image_coords="EPSG:32632", plot=False)

validation_df = build_dataset_across_images(c.df, images, c, max_pixel_padding=1, normalise=True)

# Also to load a saved classifier you can go:
# Load the classifier from the file
loaded_classifier = None
# Just use any of the names these get saved automatically
with open('Gradient Boosting.pkl', 'rb') as file:
    loaded_classifier = pickle.load(file)

ml = ML()
# validate on a single classifier
classifiers = {'Gradient Boosting Saved': loaded_classifier}
ml.classify(validation_df, loaded_classifier, 'data/classified_pixel_prediction.csv')
ml.validate_clf(validation_df, classifiers, class1_label='Sylvatica', csv_file="classifier_metrics_validation_GB_pixel_prediction_2204.csv")



