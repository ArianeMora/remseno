import osgeo
from remseno import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


coordinate_file_path = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Allenwiller_ACp_GPSdata_WGS84_epsg4326_final.csv'
c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
                   id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:4326")


from remseno import *
drone_ortho = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/allenwiller_2021_ortho_PetraRefelectance.tif'
o = Image()
o.load_image(image_path=drone_ortho)

c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:4326", plot=True)

#c.plot_on_image(image=o, band=1)
#o.plot_multi_bands(bands=[1, 2, 3], downsample=100, show_plot=True)


#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(random_state=0)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)

bands = [o.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

ml = ML()
df = ml.train_ml(clf, image=o, coords=c, image_bands=bands, validation_percent=25, test_percent=25,
                max_pixel_padding=1, normalise=False)
#df.to_csv('test_pred.csv')







