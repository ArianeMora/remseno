import osgeo
from remseno import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


coordinate_file_path = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Allenwiller_ACp_GPSdata_WGS84_epsg4326_final.csv'
c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
                   id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:4326")
<<<<<<< HEAD
<<<<<<< HEAD


from remseno import *
drone_ortho = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/allenwiller_2021_ortho_PetraRefelectance.tif'
o = Image()
o.load_image(image_path=drone_ortho)

c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:4326", plot=True)

c.plot_on_image(image=o, band=1)
o.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

bands = [o.image.read(b) for b in [1, 2, 3, 4]]

ml = ML()
df = ml.train_ml(clf, image=o, coords=c, image_bands=bands, validation_percent=30, test_percent=30,
                max_pixel_padding=3, normalise=False)
df.to_csv('test_pred.csv')




=======
>>>>>>> parent of 49bd23f (2510_working_version_Allen)

coordinate_file_path = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Allenwiller_ACp_GPSdata_WGS84_epsg4326_final.csv'
c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
                   id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:4326")

from remseno import *
drone_ortho = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/allenwiller_2021_ortho_PetraRefelectance.tif'
o = Image()
o.load_image(image_path=drone_ortho)

c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:4326", plot=True)

c.plot_on_image(image=o, band=1)
o.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)


=======


from remseno import *
drone_ortho = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/allenwiller_2021_ortho_PetraRefelectance.tif'
o = Image()
o.load_image(image_path=drone_ortho)

c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:4326", plot=True)

c.plot_on_image(image=o, band=1)
o.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)


>>>>>>> parent of 49bd23f (2510_working_version_Allen)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

bands = [o.image.read(b) for b in [1, 2, 3, 4]]

ml = ML()
df = ml.train_ml(clf, image=o, coords=c, image_bands=bands, validation_percent=30, test_percent=30,
                max_pixel_padding=3, normalise=False)
df.to_csv('test_pred.csv')
<<<<<<< HEAD
=======

>>>>>>> parent of 49bd23f (2510_working_version_Allen)


from remseno import *
drone_ortho = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/allenwiller_2021_ortho_PetraRefelectance.tif'
o1 = Image()
o1.load_image(image_path=drone_ortho)

# This is the same for image 2
drone_ortho_image_2 = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/allenwiller_2021_ortho_PetraRefelectance.tif'
o2 = Image()
o2.load_image(image_path=drone_ortho_image_2)

c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:4326", plot=True)

c.plot_on_image(image=o, band=1)
o.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)

ml = ML()
# Make a classifier any from sklearn
clf = svm.SVC(C=8.0, kernel='poly', class_weight='balanced')

images = [{'image': o1,
           'indexs': {'ndvi': get_ndvi(o1.image, 1, 2),
                      'sr': get_sr(o1.image, nir_band=1, red_band=3),
                      'band_2': o1.image.read(2),
                      'band_3': o1.image.read(3),
                      }},

          {'image': o2,  # The image read in
           'indexs': {'gndvi': get_gndvi(o1.image, 1, 2),  # Any of the indicies or you could just do a band
                      'get_normg': get_normg(o1.image, red_band=1, blue_band=2, green_band=3),
                      'ndvi': get_ndvi(o1.image, 1, 2),
                      'band_1': o1.image.read(1),
                      'band_2': o1.image.read(2),
                      'band_3': o1.image.read(3),
                      }},

          # You can keep adding more and more images here!
          ]
df = ml.train_ml_on_multiple_images(clf, images=images, coords=c, validation_percent=20, test_percent=20,
                                    max_pixel_padding=2, normalise=False)
df.to_csv('data/test_pred_multiple_images.csv')