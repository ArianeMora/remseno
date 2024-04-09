import osgeo
from remseno import *
import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
np.seterr(divide='ignore', invalid='ignore')

coordinate_file_path = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Allen_all_genotyped_trees_2024.csv'
c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
                   id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:32632")

drone_ortho = 'C:/Users/Gorde/Documents/GitHub/remseno_gk/Allen_nov.tif'
o = Image()
o.load_image(image_path=drone_ortho)

drone_ortho2 = 'C:/Users/Gorde/Documents/GitHub/remseno_gk/20230421_Allen.tif'
o2 = Image()
o2.load_image(image_path=drone_ortho2)

drone_ortho3 = 'C:/Users/Gorde/Documents/GitHub/remseno_gk/20231013_Allen.tif'
o3 = Image()
o3.load_image(image_path=drone_ortho3)

drone_ortho8 = 'C:/Users/Gorde/Documents/GitHub/remseno_gk/20230708_Allen.tif'
o8 = Image()
o8.load_image(image_path=drone_ortho8)

drone_ortho4 = 'C:/Users/Gorde/Documents/GitHub/remseno_gk/20230518_Allen.tif'
o4 = Image()
o4.load_image(image_path=drone_ortho4)

drone_ortho5 = 'C:/Users/Gorde/Documents/GitHub/remseno_gk/20230615_Allen.tif'
o5 = Image()
o5.load_image(image_path=drone_ortho5)

drone_ortho6 = 'C:/Users/Gorde/Documents/GitHub/remseno_gk/20230816_Allen.tif'
o6 = Image()
o6.load_image(image_path=drone_ortho6)

drone_ortho7 = 'C:/Users/Gorde/Documents/GitHub/remseno_gk/20230915_Allen.tif'
o7 = Image()
o7.load_image(image_path=drone_ortho7)

drone_ortho9 = 'C:/Users/Gorde/Documents/GitHub/remseno_gk/20231002_Allen.tif'
o9 = Image()
o9.load_image(image_path=drone_ortho9)




c.transform_coords(tree_coords="EPSG:32632", image_coords="EPSG:32632", plot=True)

c.plot_on_image(image=o, band=1)
o6.plot_multi_bands(bands=[1, 2, 3], downsample=10, show_plot=True)

#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(random_state=42)

# SVM with linear kernel
#clf = svm.SVC(kernel='linear')

clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.2,
    max_depth=3,random_state=42
)

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=5, random_state=42)


# Function to get all indices
def get_all_planetscope(img):
    #nitian = get_nitian(image=img, r_edge=7, blue_band=2)
    ndvi = get_ndvi(image=img, red_band=6, nir_band=8)
    sr = get_sr(image=img, red_band=6, nir_band=8)
    #tvi = get_tvi(image=img, red_band=6, rededge_band=7, green_band=4)
    #gi = get_gi(image=img, red_band=6, green_band=4)
    #gndvi = get_gndvi(image=img, green_band=4, nir_band=8)
    pri = get_pri(image=img, green_band=4, greeni_band=3)
    #osavi = get_osavi(image=img, red_band=6, nir_band=8)
    tcari = get_tcari(image=img, rededge_band=7, greeni_band=3, red_band=6)
    redge = get_redge(image=img, nir_band=8, green=4, r_edge=7)
    #redge2 = get_redge2(image=img, red_band=6, green=4, r_edge=7)
    #siredge = get_siredge(image=img, red_band=6, r_edge=7)
    #normg = get_normg(image=img, red_band=6, green_band=4, blue_band=2)
    schl = get_schl(image=img, red_band=6, rededge_band=7, nir_band=8)
    schlcar = get_schlcar(image=img, red_band=6, greeni_band=3)
    return ndvi, sr, pri, tcari, redge, schl, schlcar

# Get all indices
indices = get_all_planetscope(o.image)

#to bands add the indices calculated from the bands in indices.py
bands = list(indices) + [o2.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]] + [o3.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]] + [o4.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]] +[o5.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]]+[o6.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]]+[o7.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]]+ [o8.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]]+ [o9.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]]
#bands = [o9.image.read(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]]

ml = ML()
df = ml.train_ml(clf, image=o, coords=c, image_bands=bands, validation_percent=30, test_percent=30,
                max_pixel_padding=2, normalise=False)

test_df


#df.to_csv('test_pred.csv')


# Feature importance
#importances = clf.feature_importances_
#indices = np.argsort(importances)[::-1]

# Print the feature ranking
#print("Feature ranking:")
#for f in range(df.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
# Ensure the lengths match for the x-axis and importances
#plt.bar(range(len(importances)), importances[indices], color="r", align="center")
#plt.xticks(range(len(importances)), indices)
#plt.xlim([-1, len(importances)])
#plt.show()