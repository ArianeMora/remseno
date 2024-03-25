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
Just dataset creation, for a range of applications. Someones ya may want to join images. Sometimes just get the coords.
Sometimes normalising... This just combines & does that sort of thing.
"""

import numpy as np
import pandas as pd


def build_dataset_from_image(df, image, coords, image_bands, max_pixel_padding=1, normalise=True):
    """
    Builds a dataset from a single image and coordinate file.
    """
    xs = df[coords.x_col].values
    ys = df[coords.y_col].values
    rows = []
    # Image bands is something like below, we leave it up to the user to define i.e. could be indicies
    classes = df[coords.label_col].values
    band_labels = [f'band_{band_i}' for band_i, c in enumerate(image_bands)]

    for i, tid in enumerate(df[coords.id_col].values):
        y, x = image.image.index(xs[i], ys[i])
        # Now for each bounding area make a training point
        bb = coords.build_circle_from_centre_point(x, y, max_pixel_padding)
        for xy in bb:
            # We now build the feature set which is
            data_row = [tid, classes[i], xy[0], xy[1]]
            for band_i, image_band in image_bands.items():
                # Here we are extracting the feature i.e. the value from the image bands we're interested in
                data_row.append(image_band[xy[1], xy[0]])
            rows.append(data_row)
    # Now create a dataframe from this
    train_df = pd.DataFrame(rows, columns=[coords.id_col, coords.label_col, coords.x_col, coords.y_col] +
                                          band_labels)
    # If normalise we normalise each row
    if normalise:
        train_df = train_df.fillna(0)

        # Use a standard scaler on each row
        for b in band_labels:
            vals = train_df[b]
            train_df[b] = (vals - np.mean(vals))/np.std(vals)  # Just do the usual standard scaling!
        train_df = train_df.fillna(0)

    return train_df


def build_dataset_across_images(df, images, coords, max_pixel_padding=1, normalise=True):
    """
    This basically just concatenates the features across the different features for a single set of trees.
    For example, you have mulitple time points for a single forest and you want to capture the temporal
    variation since that is important for species classification. In that case you'd use this function.
    """
    xs = df[coords.x_col].values
    ys = df[coords.y_col].values
    rows = []
    # Image bands is something like below, we leave it up to the user to define i.e. could be indicies
    classes = df[coords.label_col].values
    for i, tid in enumerate(df[coords.id_col].values):
        # We now build the feature set which is
        data_row = [tid, classes[i]]
        for image_values in images:
            image = image_values['image']
            y, x = image.image.index(xs[i], ys[i])
            # Now for each bounding area make a training point
            bb = coords.build_circle_from_centre_point(x, y, max_pixel_padding)
            for xy in bb:
                for label, image_band in image_values['indexs'].items():
                    # Here we are extracting the feature i.e. the value from the image bands we're interested in
                    data_row.append(image_band[xy[1], xy[0]])
        rows.append(data_row)
    # Now create a dataframe from this
    train_df = pd.DataFrame(rows)
    cols = [c for i, c in enumerate(train_df.columns) if i > 1]
    # If normalise we normalise each row
    if normalise:
        # Use a standard scaler on each column
        for b in cols:
            vals = train_df[b]
            train_df[b] = (vals - np.mean(vals))/np.std(vals)  # Just do the usual standard scaling!
    cols = [coords.id_col, coords.label_col] + [c for i, c in enumerate(train_df.columns) if i > 1]
    train_df.columns = cols
    return train_df


def build_dataset_from_multiple_images_coords(location_df, images, coords, max_pixel_padding=1,
                                              normalise=False):
    """
    This builds a dataset from multiple images with multiple coordinate files, basically just concats them
    together so that the dataset can be larger. i.e. you have multiple training sets and you want to build
    a larger one across different images.
    """
    train_df = pd.DataFrame()
    for i in range(0, len(images) - 1):
        image = images[i]
        tmp_df, training_cols = build_dataset_from_image(location_df, image['image'], coords[i],
                                                              image['indexs'], max_pixel_padding, normalise)
        train_df = pd.concat([train_df, tmp_df])
    train_df = train_df.fillna(0)
    return train_df

