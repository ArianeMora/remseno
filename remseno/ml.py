###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

"""
Machine learning component of the project.
"""
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ML:

    def __init__(self):
        print('ML go brr')

    def kmeans(self, image, bands=None, k=5):
        """
        Perform kmeans clustering on an image/ortho.
        :param image: either satelite image or an ortho
        :param k: number of clusters
        :return:
        """
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto") #.fit(X)
        if bands is None:
            # Use all bands by default
            bands = image.get_bands()
        dataset = []
        for band in bands:
            dataset.append(image.get_band(band))
        # Convert to numpy array

    def build_train_df(self, image, bands: list, coords, max_pixel_padding=2):
        df = coords.df
        # Build training dataset using the different channels in the tiff
        train_df = pd.DataFrame()
        classes =
        for i, tid in enumerate(df[coords.id_col].values):
            y, x = image.image.index(xs[i], ys[i])
            # Now for each bounding area make a training point
            bb = coords.build_circle_from_centre_point(x, y, max_pixel_padding)
            for xy in bb:
                # We now build the feature set which is
                data_row = [tid, classes[i], xy[0], xy[1]]
                for image_band in bands:
                    try:
                        # Here we are extracting the feature i.e. the value from the image bands we're interested in
                        data_row.append(image_band[xy[0], xy[1]])
                    except:
                        print(tid, 'ERROR')
                        data_row.append(0)
                rows.append(data_row)
            # Now create a dataframe from this
        train_df = pd.DataFrame(rows,
                                columns=[coords.id_col, coords.binary_label, coords.x_col, coords.y_col] + band_labels)
        return train_df, band_labels

        return train_df

    def binary_classifier(self, image, coords, bands: list, valid_size=0.25):
        df = coords.df
        band1 = image.image.read(1)

        # Build training dataset using the different channels in the tiff
        for pixel_padding in range(1, 10):
            train_df = pd.DataFrame()  # df[[id_col, binary_label, label_col]].copy() # Make a copy of the cols we're interested in

            for band in bands:
                pixel_values = []
                ids = []
                labels = []
                bin_labels = []
                curr_band = image.image.read(band)
                # Also do the x, y values
                x_new = []
                y_new = []
                # Now for each tree we want to get the pixel values
                y_vals = df[coords.y_col].values
                id_vals, bin_vals, lbl_vals = df[coords.id_col].values, df[coords.binary_label].values, df[coords.label_col].values
                for i, x in enumerate(df[coords.x_col].values):
                    y, x = image.image.index(x, y_vals[i])
                    for xj in range(-pixel_padding, pixel_padding):
                        for yj in range(-pixel_padding, pixel_padding):
                            pixel_tree = curr_band[y + yj, x + xj]  # Check the direction!!!
                            pixel_values.append(pixel_tree)
                            ids.append(id_vals[i])
                            bin_labels.append(bin_vals[i])
                            labels.append(lbl_vals[i])
                            x_new.append(x + xj)
                            y_new.append(y + yj)

                train_df[coords.id_col] = ids
                train_df[coords.binary_label] = bin_labels
                train_df[coords.label_col] = labels
                train_df[f'band_{band}'] = pixel_values
                train_df[f'x_new'] = x_new
                train_df[f'y_new'] = y_new

            # Subsample the min number
            sampled_df = pd.DataFrame()
            num_samples = min(train_df[coords.binary_label].value_counts())
            for x in set(train_df[coords.binary_label].values):
                # Take a sample from the df and add it to the sampled DF
                subsample = train_df[train_df[coords.binary_label] == x].sample(num_samples)
                sampled_df = pd.concat([sampled_df, subsample], ignore_index=True)

            # Train ML model
            train_cols = [f'band_{band}' for band in bands]
            # Make an even subsample

            X = sampled_df[train_cols].values
            y = sampled_df[coords.binary_label].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valid_size,
                                                                random_state=0)
            # Get the pixels from the orthomosaic
            scaler = preprocessing.StandardScaler().fit(X_train)
            clf = svm.SVC(C=8).fit(X_train, y_train)
            clf.score(X_test, y_test)
            test_score = clf.score(X_test, y_test)
            y_pred = clf.predict(X_test)

            # Do the visualisation of the prediction
            from sklearn.metrics import ConfusionMatrixDisplay
            from sklearn.metrics import confusion_matrix

            fig, axs = plt.subplots(1, 3, figsize=(15, 4))

            fig0 = axs[0]
            fig1 = axs[1]
            fig2 = axs[2]

            # Make the 3 subplots
            confmat = confusion_matrix(y_test, y_pred)
            sns.heatmap(confmat, annot=True, ax=fig2, cmap='viridis')
            fig2.set_title('Test acc.')

            fig0.imshow(band1, cmap='pink')
            fig0.set_title(f'True labels, test acc: {int(test_score * 100)}%', fontweight="bold")

            ys = df[coords.y_col].values
            # Do the prediction
            df[f'colour'] = ['blue' if c == 0 else 'red' for c in df['binary_label'].values]
            colours = df[f'colour'].values

            for i, x in enumerate(df[coords.x_col].values):
                try:
                    y, x = image.image.index(x, ys[i])
                    fig0.scatter(x, y, c=colours[i], s=10)
                except:
                    print(i, x)

            train_df['predicted_label'] = clf.predict(
                train_df[train_cols].values)  # Get the prediction capacity as the average of the pixels
            grped_df = train_df.groupby(coords.id_col)

            # Map the ID to the predicted label
            id_value_map = {}
            correct = 0
            incorrect = 0
            for gid, g_df in grped_df:
                # Get the most common colour
                true_label = g_df[coords.binary_label].values[0]
                # Now check what the average prediction was
                pred_correct = len(g_df[g_df['predicted_label'] == true_label])
                pred_incorrect = len(g_df[g_df['predicted_label'] != true_label])
                # average prediction is going to just be the most common prediction
                pred_value = None
                if pred_correct > pred_incorrect:
                    pred_value = true_label
                    correct += 1
                else:
                    # Only works for the binary class problem
                    pred_value = 0 if true_label == 1 else 1
                    incorrect += 1
                id_value_map[gid] = {'pred': pred_value, 'pred_prob': pred_correct / (pred_correct + pred_incorrect)}

            total_prob = correct / (correct + incorrect)
            preds = []
            pred_probs = []

            for tid in df[coords.id_col].values:
                preds.append(id_value_map[tid]['pred'])
                pred_probs.append(id_value_map[tid]['pred_prob'])

            df['pred'] = preds
            df['pred_probs'] = pred_probs

            fig1.imshow(band1, cmap='BuGn')
            fig1.set_title(f'Predicted: {int(total_prob * 100)}%', fontweight="bold")

            ys = df[coords.y_col].values

            # Do the prediction
            df[f'colour'] = ['blue' if c == 0 else 'red' for c in df['pred'].values]
            colours = df[f'colour'].values

            for i, x in enumerate(df[coords.x_col].values):
                try:
                    y, x = image.image.index(x, ys[i])
                    fig1.scatter(x, y, c=colours[i], s=10)
                except:
                    print(i, x)

            plt.title(f'Number of pixels bounding: {pixel_padding}', fontweight="bold")
            plt.savefig(f'../img/image_{pixel_padding}.png', dpi=300)
            plt.show()
