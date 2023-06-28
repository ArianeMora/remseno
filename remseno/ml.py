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
import math
import os

import rasterio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pickle


class ML:

    def __init__(self):
        print('ML go brr')
        self.train_samples = None
        self.validation_df = None
        self.train_df = None
        self.clf = None

    def save(self, output_dir, label=''):
        pickle.dumps(os.path.join(output_dir, f'classifier_{label}.pkl'))
        self.train_df.to_csv(os.path.join(output_dir, f'trainDF_{label}.csv'), index=False)
        self.validation_df.to_csv(os.path.join(output_dir, f'validDF_{label}.csv'), index=False)

    def build_train_df(self, df, image, coords, bands, max_pixel_padding=2,
                       band_labels=None, normalise=False):
        """
        Build a training dataframe from image and coordinate files!

        :param image:
        :param coords:
        :param image_bands:
        :param max_pixel_padding:
        :param band_labels:
        :return:
        """
        band_labels = band_labels if band_labels else [f'b{i}' for i in range(0, len(bands))]
        xs = df[coords.x_col].values
        ys = df[coords.y_col].values
        rows = []
        # Image bands is something like below, we leave it up to the user to define i.e. could be indicies
        image_bands = []
        for band in bands: # Always normalise so that it is easier for
            normed = image.image.read(band)
            if normalise:
                image_bands.append(rasterio.plot.adjust_band(normed))
            else:
                image_bands.append(normed) # #normed-np.mean(normed)) #(normed - np.min(normed)) / (np.max(normed) - np.min(normed)))

        classes = df[coords.binary_label].values
        for i, tid in enumerate(df[coords.id_col].values):
            y, x = image.image.index(xs[i], ys[i])
            # Now for each bounding area make a training point
            bb = coords.build_circle_from_centre_point(x, y, max_pixel_padding)
            for xy in bb:
                # We now build the feature set which is
                data_row = [tid, classes[i], xy[0], xy[1]]
                for image_band in image_bands:
                    # Here we are extracting the feature i.e. the value from the image bands we're interested in
                    data_row.append(image_band[xy[1], xy[0]])
                rows.append(data_row)
        # Now create a dataframe from this
        train_df = pd.DataFrame(rows, columns=[coords.id_col, coords.binary_label, coords.x_col, coords.y_col] + band_labels)
        return train_df, band_labels

    def train_ml(self, clf, image, coords, bands, validation_percent=20, test_percent=20,
                 max_pixel_padding=2, normalise=False):
        """
        Train a ML classifier for the image and coords.

        :return: VAE or something
        """
        # Build training DF from the coords
        # First we want to hold some trees out, so we select a random sample from the dataset
        valid_df = coords.df.sample(math.ceil(len(coords.df)*(validation_percent/100)))
        # Now use the other as the dataframe
        df = coords.df[~coords.df[coords.id_col].isin(list(valid_df[coords.id_col].values))]
        df, training_cols = self.build_train_df(df, image, coords, bands, max_pixel_padding=max_pixel_padding,
                                                normalise=normalise)
        self.train_df = df
        self.valid_df, training_cols = self.build_train_df(valid_df, image, coords, bands,
                                                           max_pixel_padding=max_pixel_padding, normalise=normalise)

        X = df[training_cols].values
        y = df[coords.binary_label]
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent/100,
                                                            random_state=18)
        # Get the pixels from the orthomosaic
        clf = clf.fit(X_train, y_train)
        clf.score(X_test, y_test)
        test_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        print(test_score, y_pred)
        # Then also do the final ones in the validation df i.e. unseen trees
        X = self.valid_df[training_cols].values
        y = self.valid_df[coords.binary_label]
        print(clf.score(X, y))
        print(balanced_accuracy_score(y, clf.predict(X)))
        print(len(self.valid_df), len(self.train_df))
        # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
        # Solve for w3 (z)
        z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]
        Y = y
        tmp = np.linspace(0, 1, 30)
        x, y = np.meshgrid(tmp, tmp)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(x, y, z(x, y))
        ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], 'ob')
        ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
        plt.show()
        # Add the predicted label in for each
        self.train_df['predicted_label'] = clf.predict(self.train_df[training_cols].values)
        self.valid_df['predicted_label'] = clf.predict(self.valid_df[training_cols].values)
        self.clf = clf # Save to the model
        return self.get_overall_tree_pred(coords.df, coords, self.train_df, self.valid_df)

    def group_results(self, train_df, coords, correct, incorrect, id_value_map, data_type):
        """
        Group results.

        :param train_df:
        :param coords:
        :param correct:
        :param incorrect:
        :param id_value_map:
        :return:
        """
        grped_df = train_df.groupby(coords.id_col)
        for gid, g_df in grped_df:
            # Get the most common colour
            true_label = g_df[coords.binary_label].values[0]
            # Now check what the average prediction was
            pred_correct = len(g_df[g_df['predicted_label'] == true_label])
            pred_incorrect = len(g_df[g_df['predicted_label'] != true_label])
            # average prediction is going to just be the most common prediction
            if pred_correct > pred_incorrect:
                pred_value = true_label
                correct += 1
            else:
                # Only works for the binary class problem
                pred_value = 0 if true_label == 1 else 1
                incorrect += 1
            overall_pred = 'correct' if correct > incorrect else 'incorrect'
            id_value_map[gid] = {'pred': pred_value, 'pred_prob': pred_correct / (pred_correct + pred_incorrect),
                                 'type': data_type, 'overall_pred': overall_pred}
        return id_value_map, correct, incorrect

    def get_overall_tree_pred(self, df, coords, train_df, valid_df):
        """
        :return:
        """
        # Do it for both the training and the validation dataframe
        # Map the ID to the predicted label
        id_value_map, correct, incorrect = {}, 0, 0
        id_value_map, correct, incorrect = self.group_results(train_df, coords, correct, incorrect, id_value_map, 'train')
        id_value_map, correct, incorrect = self.group_results(valid_df, coords, correct, incorrect, id_value_map, 'valid')
        # Do the same for validation
        total_prob = correct / (correct + incorrect)
        print(total_prob)
        preds = []
        pred_probs = []
        data_types = []
        overall_pred = []
        for tid in df[coords.id_col].values:
            preds.append(id_value_map[tid]['pred'])
            pred_probs.append(id_value_map[tid]['pred_prob'])
            data_types.append(id_value_map[tid]['type'])
            overall_pred.append(id_value_map[tid]['overall_pred'])

        df['pred'] = preds
        df['pred_probs'] = pred_probs
        df['data_type'] = data_types
        df['overall_pred'] = overall_pred
        return df