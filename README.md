# remseno

A package for predicting tree species using remote sensing data (either drone orthomosaic or satellite data).

# install

```
pip install dist/remseno.
```

# quick start
You need: a coordinate file (which has the tree ID, tree label (i.e. class), X, and Y coordinates, and the coordinate reference system).

## example with RBG data from drone footage

### load in coordinate file
```python
from remseno import *
coordinate_file_path = f'./data/dryad_trees/QGIS/Annotations.csv' 
c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
                   id_col='id', sep=',', class1='RedCedar', class2='Pine', crs="EPSG:4326")

```
### load in ortho or satellite image data

```python
from remseno import *
drone_ortho = '../data/dryad_trees/Stitch_Image/20190518_pasture_100ft_RGB_GCPs_Forest.tif'
o = Image()
o.load_image(image_path=drone_ortho)
```

Once the image has been loaded it will print out the coordinate reference system, if that differs to the one used in the 
image you'll need to change coordinates

### optionally: update coordinate system
```python
c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32614", plot=True)
```
### run through the pipeline

1. Plot your coords on the image as points
```python
c.plot_on_image(image=o, band=1)
```
2. Plot bounding boxes (in metres) around your coordinates
   (ToDo)
3. Plot multiple bands
downsample = reduce quality of image this makes it faster!
```python
o.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)
```
4. Plot each class as a subset individually (ToDo)
6. Perform classification
First you need to pick a classifier, this depends on your problem and interests, anything from sklearn can be passed:
e.g. `https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html`
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
```
Then you just pass that to the classifier function and you get back a dataframe with your predictions based on the pixels:
The `max_pixel_padding` tells you how many pixels will be included as training (this is the circle around each central location).
```python
ml = ML()
ml_df = ml.train_ml(clf, image=o, bands=[1, 2, 3], coords=c, normalise=True, max_pixel_padding=1)
ml_df
```

## example with hyperspectral satellite data
This is basically the same as above, except we can create indicies and masks.


### load in coordinate file
```python
from remseno import *
coordinate_file_path = '../data/tallo/planetscope/planetscope_test.csv'
c = Coords(coordinate_file_path, x_col='latitude', y_col='longitude', label_col='tree_id',
           id_col='tree_id', sep=',', class1='T_498824', class2='T_498824', crs="EPSG:4326")

```
### load in ortho or satellite image data

```python
from remseno import *
sat = '../data/tallo/planetscope/sat_data/dab9fa59-4a98-4319-ad5f-acea23bc6feb/PSScene/20230113_230924_81_241e_3B_AnalyticMS_SR_8b_clip.tif'
o = Image()
o.load_image(image_path=sat)
```

Once the image has been loaded it will print out the coordinate reference system, if that differs to the one used in the 
image you'll need to change coordinates

### optionally: update coordinate system
```python
c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32755", plot=True)
```
### run through the pipeline

1. Plot your coords on the image as points
```python
c.plot_on_image(image=o, band=1)
```
2. Plot bounding boxes (in metres) around your coordinates
   (ToDo)
3. Plot multiple bands
downsample = reduce quality of image this makes it faster!
```python
o.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)
```
4. Plot indicies
```python
ndvi = get_ndvi(image=o.image, red_band=6, nir_band=8)
o.plot_idx(ndvi)
plt.show()
```
5. Make a mask on an index
```python
ax = o.plot_rbg()
ndvi = get_ndvi(image=o.image, red_band=6, nir_band=8)
mask = o.mask_on_index(ndvi, index_cutoff=0.75)  # This is the filter for what to mask so anything < 0.75 will be masked
plt.imshow(mask*ndvi)
plt.show()
```
6. Save as rbg for visualisation in other programs
```python
o.write_as_rbg('rbg_version.tif')
```
7. Perform classification
First you need to pick a classifier, this depends on your problem and interests, anything from sklearn can be passed:
e.g. `https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html`
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
```
Then you just pass that to the classifier function and you get back a dataframe with your predictions based on the pixels:
The `max_pixel_padding` tells you how many pixels will be included as training (this is the circle around each central location).
```python
ml = ML()
ml_df = ml.train_ml(clf, image=o, bands=[1, 2, 3], coords=c, normalise=True, max_pixel_padding=1)
ml_df
```