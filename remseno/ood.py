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
Create a distribution from the spectral data so that we know what is "out of distribution" and what is "in distribution"
"""
import pandas as pd
import numpy as np
from scivae import *


class OOD:

    def __init__(self, image, coords):
        self.vae = None
        self.image = image
        self.coords = coords
        self.config = {'scale_data': True,
                          # Whether to min max scale your data VAEs work best when data is pre-normalised & outliers removed for trainiing
                          'batch_norm': True,
                          'loss': {'loss_type': 'mse',  # mean squared error
                                   'distance_metric': 'mmd',  # Maximum mean discrepency (can use kl but it works worse)
                                   'mmd_weight': 1},
                          # Weight of mmd vs mse - basically having this > 1 will weigh making it normally distributed higher
                          # and making it < 1 will make reconstruction better.
                          'encoding': {'layers': [{'num_nodes': 8, 'activation_fn': 'selu'},  # First layer of encoding
                                                  {'num_nodes': 4, 'activation_fn': 'selu'}]},  # Second layer of encoding
                          'decoding': {'layers': [{'num_nodes': 4, 'activation_fn': 'selu'},  # First layer of decoding
                                                  {'num_nodes': 8, 'activation_fn': 'selu'}]},  # Second layer of decoding
                          'latent': {'num_nodes': 1},
                          'optimiser': {'params': {}, 'name': 'adam'}}  # Empty params means use default

    def load_saved_vae(self, weight_file_path='model_weights.h5', optimizer_file_path='model_optimiser.json',
             config_json='config.json', load_data=False, data_filename='data.csv'):

        self.vae = VAE([[0]], [[0]], [[0]], self.config, '')
        self.vae.load(weight_file_path, optimizer_file_path, config_json, load_data, data_filename)

    def build_train_df(self, image, coords, bands, max_pixel_padding=2, band_labels=None):
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
        df = coords.df
        xs = df[coords.x_col].values
        ys = df[coords.y_col].values
        rows = []
        # Image bands is something like below, we leave it up to the user to define i.e. could be indicies
        #image_bands = [image.ortho.get_band(b) for b in bands]
        classes = df[coords.binary_label].values
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
        train_df = pd.DataFrame(rows, columns=[coords.id_col, coords.binary_label, coords.x_col, coords.y_col] + band_labels)
        return train_df, band_labels

    def train_ood(self, image, coords, bands, config=None, max_pixel_padding=2, band_labels=None):
        """
        Train an OOD classifier (i.e. a VAE) to check if a pixel is out of expected distribution.

        :return: VAE or something
        """
        # Build training DF
        df, training_cols = self.build_train_df(image, coords, bands, max_pixel_padding=2)
        numpy_array = df[training_cols].values
        # Train model
        config = self.config if config is None else config
        vae_mse = VAE(numpy_array, numpy_array, df[coords.binary_label].values, config, 'vae_label')
        # Set batch size and number of epochs
        vae_mse.encode('default', epochs=100, batch_size=50, early_stop=True)
        encoded_data_vae_mse = vae_mse.get_encoded_data()
        # Save model
        vae_mse.save()
        self.vae = vae_mse
        # Also save to instance so we can use it for this iteration
        return encoded_data_vae_mse

    def classify_ood(self, data_np: np.array, cutoff=1.0):
        """
        Classify whether the points in a df are ood
        :param df:
        :return:
        """
        # Assume the DF is already created
        # Get the vae
        pred = self.vae.encode_new_data(data_np, encoding_type="z", scale=True)
        # Now we want to check if it is 2 SD > or < than mean
        # VAEs are standard normal so if it is +- 2
        ood = []
        for vae_value in pred:  # Maybe make neater if you're feeling nice
            vae_value = np.max(abs(vae_value))  # Calculate the greatest deviation
            if vae_value > cutoff:
                ood.append(True)
            else:
                ood.append(False)
        return ood, pred