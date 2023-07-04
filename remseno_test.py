from remseno import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


coordinate_file_path = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Waeldi_Adults_genotyped2.csv'
c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
                   id_col='id', sep=',', class1='Sylvatica', class2='Orientalis', crs="EPSG:21781")


from remseno import *
drone_ortho = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/waeldi_2022_multispectral_noreflectance.tif'
o = Image()
o.load_image(image_path=drone_ortho)

c.transform_coords(tree_coords="EPSG:21781", image_coords="EPSG:4326", plot=True)

c.plot_on_image(image=o, band=1)
o.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

ml = ML()
ml_df = ml.train_ml(clf, image=o, bands=[1, 2, 3], coords=c, normalise=True, max_pixel_padding=1)
ml_df.to_csv('C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/nov.csv')







