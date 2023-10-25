import osgeo
from remseno import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


coordinate_file_path = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Allenwiller_ACp_Coordinates_UTM32N_32632.csv'
c = Coords(coordinate_file_path, x_col='X', y_col='Y', label_col='class',
                   id_col='id', sep=',', class1='Sylvatica', class2='Orientals', crs="EPSG:4326")


from remseno import *
drone_ortho = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Allen_june.tif'
o = Image()
o.load_image(image_path=drone_ortho)

c.transform_coords(tree_coords="EPSG:32632", image_coords="EPSG:32632", plot=True)

# After you have performed the transformation, you can save the DataFrame to a new CSV file.
#output_file_path = 'C:/Users/Gorde/Documents/GitHub/remhybmon/data/public_data/Waeldi_Adults_transformed_32632.csv'  # Specify the path for the output file.

# Assuming that 'c.df' contains the transformed data
#c.df.to_csv(output_file_path, index=False)  # Save the DataFrame to a CSV file, excluding the index.

# Optionally, you can print a message to confirm that the file has been saved.
#print(f"Transformed coordinates saved to {output_file_path}")

c.plot_on_image(image=o, band=1)
o.plot_multi_bands(bands=[1, 2, 3], downsample=5, show_plot=True)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)

bands = [o.image.read(b) for b in [2, 3, 4, 5, 6, 7]]

ml = ML()
df = ml.train_ml(clf, image=o, coords=c, image_bands=bands, validation_percent=40, test_percent=40,
                max_pixel_padding=2, normalise=True)








