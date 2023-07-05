from remseno import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


coordinate_file_path = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Waeldi_Adults_genotyped2.csv'
c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
                   id_col='id', sep=',', class1='Sylvatica', class2='Orientalis', crs="EPSG:21781")


from remseno import *
drone_ortho = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/waldi_july.tif'
o = Image()
o.load_image(image_path=drone_ortho)

c.transform_coords(tree_coords="EPSG:21781", image_coords="EPSG:32632", plot=True)

c.plot_on_image(image=o, band=1)
o.plot_multi_bands(bands=[1, 2, 3], downsample=1, show_plot=True)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

bands = [o.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]]

ml = ML()
df = ml.train_ml(clf, image=o, coords=c, image_bands=bands, validation_percent=20, test_percent=20,
                max_pixel_padding=1, normalise=False)
df.to_csv('test_pred.csv')







