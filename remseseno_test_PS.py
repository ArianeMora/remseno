import osgeo
from remseno import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


coordinate_file_path = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Waeldi_Adults_genotyped.csv'
c = Coords(coordinate_file_path, x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:21781")

c.transform_coords(tree_coords="EPSG:21781", image_coords="EPSG:4326", plot=True)












