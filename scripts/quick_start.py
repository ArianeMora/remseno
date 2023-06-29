from remseno import *
from remseno.indices import *

coordinate_file_path = f'../data/dryad_trees/QGIS/Annotations.csv'
c = Coords(coordinate_file_path, x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1='RedCedar', class2='Pine', crs="EPSG:4326")
drone_ortho = '../data/dryad_trees/Stitch_Image/20190518_pasture_100ft_RGB_GCPs_Forest.tif'
o = Image()
o.load_image(image_path=drone_ortho)
c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32614", plot=True)

c.plot_on_image(image=o, band=1)

o.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

ml = ML()
ml_df = ml.train_ml(clf, image=o, bands=[1, 2, 3], coords=c, normalise=True, max_pixel_padding=1)

from remseno import *
coordinate_file_path = '../data/tallo/planetscope/planetscope_test.csv'
c = Coords(coordinate_file_path, x_col='latitude', y_col='longitude', label_col='tree_id',
           id_col='tree_id', sep=',', class1='T_498824', class2='T_498824', crs="EPSG:4326")

from remseno import *
sat = '../data/tallo/planetscope/sat_data/dab9fa59-4a98-4319-ad5f-acea23bc6feb/PSScene/20230113_230924_81_241e_3B_AnalyticMS_SR_8b_clip.tif'
o = Image()
o.load_image(image_path=sat)

c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32755", plot=True)

c.plot_on_image(image=o, band=1)

o.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)

ndvi = get_ndvi(image=o.image, red_band=6, nir_band=8)
o.plot_idx(ndvi)
plt.show()

ax = o.plot_rbg()
ndvi = get_ndvi(image=o.image, red_band=6, nir_band=8)
mask = o.mask_on_index(ndvi, index_cutoff=0.75)  # This is the filter for what to mask so anything < 0.75 will be masked
plt.imshow(mask*ndvi)
plt.show()

o.write_as_rbg('rbg_version.tif')

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

ml = ML()
ml_df = ml.train_ml(clf, image=o, bands=[1, 2, 3], coords=c, normalise=True, max_pixel_padding=1)
ml_df