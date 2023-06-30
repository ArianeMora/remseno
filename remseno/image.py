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
from tqdm import tqdm
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import rasterio.warp
from rasterio.crs import CRS

"""
Generates the orthomosaic using Open drone map.


Inputs:
1. flight_dirs: a list of folders which have subfolders with .tif files
2. output_dir: where you want all the files to be copied to

Outputs:
1. a copy of all your images renamed
2. a csv named image_name_map in the output folder which has the old path and new path for each image
"""
from remseno.base import Remsenso
import os
import numpy as np


# Normalize bands into 0.0 - 1.0 scale
def normalise(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


class Image(Remsenso):

    def __init__(self, name=''):
        super().__init__()
        self.image = None
        self.crs = None
        self.name = name

    def get_bands(self):
        """
        Get all the bands in an image
        :return:
        """
        return self.image.count

    def get_pix_to_m(self):
        """
        Return x, y, units, pixel to m (or feet) resolution
        :return:
        """
        # https://rasterio.readthedocs.io/en/stable/quickstart.html#dataset-attributes
        crs = self.image.crs
        x1 = self.image.transform * (1, 1)
        x0 = self.image.transform * (0, 0)
        if crs.linear_units_factor == 'metre':
            return (x1[0] - x0[0])*100, (x1[1] - x0[1])*100, crs.linear_units_factor
        else:
            return (x1[0] - x0[0])*100, (x1[1] - x0[1]), crs.linear_units_factor

    def get_band(self, band_index=int):
        """
        Get the band from the orthomosaic.

        :param band_index:
        :return:
        """
        return self.image.read(band_index)

    def rename_ortho_photos(self, flight_dir: str, output_dir: str):
        """
        # 1. Rename images and put them in a single folder

        **NOTE**: The `.tif` files have the band annotated in the postfix *i.e.* `img_008_1.tif` means it is using
        band `1`. Thus we need to add a prefix for the photo rather than a postfix.
        This is going to just be randomly generated and saved to a csv for reproducibility given this is the most
         system agnostic method.

        :param flight_dir: directory that contains subfolders of the tif files
        :param output_dir: directory to save the renamed images to
        :return:
        """

        # Makes the subfolder
        self.u.dp(['Making images folder in:', output_dir])
        image_output = os.path.join(output_dir, "images")
        self.run_cmd(f'mkdir {image_output}')

        self.u.dp(['Running image copying to:', output_dir])
        i = 0
        fails = 0
        # Makes the image mapping file
        with open(f'{output_dir}image_name_map.csv', 'w+') as fout:
            fout.write(f'OriginalPath,NewPath\n')
            try:
                # iterate through all subfolders until we find tifs and then copy those (labelling them by their path)
                for root, dirs, files in os.walk(flight_dir):
                    tifs = False
                    self.u.dp(['Copying files in folder:', root])

                    for name in tqdm(files):
                        if '.tif' in name:  # Does this for all tif files
                            new_name = f'F{i}{name}'
                            # Copy image --> use os.path to make it system agnostic
                            self.run_cmd(f'cp {os.path.join(root, name)} {os.path.join(image_output, new_name)}')
                            # If a success lets add the name to our csv file
                            fout.write(f'{os.path.join(root, name)},{os.path.join(image_output, new_name)}\n')
                            i += 1
            except Exception as e:
                self.u.err_p(["FAILED", flight_dir, '\n', str(e)])
                # If a failure write out the error
                fout.write(f'{os.path.join(root, name)},FAIL: {str(e)}\n')
                fails += 1

        self.u.dp(['Successfully copied: ', i, ' files to: ', f'{image_output}'])
        self.u.dp(['Failed on: ', fails, ' files'])

    def run_ortho(self, params: str, input_dir: str):
        """
        ## 2. Running Orthomosaic

        Here we run the orthomosaic using open drone map (ODM). Documentation can be found here:
        https://docs.opendronemap.org/.

        :param params: parameters for the ODM e.g. --radiometric-calibration camera+sun --feature-quality high
        :param input_dir: directory with the images in it
        :return:
        """
        self.u.dp(['Printing docker command using your parameters:\n', params,
                  '\nWith images data in folder: ', input_dir,
                   '\nThis will take a long time. For a laptop with 5k images it will probably die.',
                   '\nWould suggest running overnight or with low quality for a first run.'])

        self.run_cmd(f'docker run -ti --rm -v {input_dir}:/datasets opendronemap/odm {params} --project-path '
                     f'/datasets /datasets')

        self.u.warn_p(['You will need to copy and paste that command into your terminal now!'])

    def plot_idx(self, band, ax=None, show_plot=False, downsample=None, cmap='pink', roi= None):
        """
        Plot specific bands or calculated index (as below but just uses the band already calcualted)

        :param band:
        :param show_plot: whether or not to show th eplot
        :param ax:
        :return: ax
        """
        # Plot the first band
        if ax is None:
            fig, ax = plt.subplots()
        if roi:
            band = band[roi['y1']: roi['y2'], roi['x1']: roi['x2']]
        if downsample:
            band = band[::downsample, ::downsample]
        ax.imshow(band, cmap=cmap)
        ax.set_title(f'{self.name}')
        if show_plot:
            plt.show()
        return ax

    def plot_multi_bands(self, bands: list, title='', ax=None, show_plot=False, downsample=None, roi=None):
        """
        Plots multiple
        :param bands: a list of bands to plot
        :param ax:
        :param show_plot:
        :return:
        """
        if ax is None:
            fig, ax = plt.subplots()
        # Convert to numpy arrays
        img_bands = []
        for b in bands:
            ds = self.image.read(b)
            if roi:
                ds = ds[roi['y1']:roi['y2'], roi['x1']:roi['x2']]
            if downsample:
                ds = ds[::downsample, ::downsample]
            img_bands.append(normalise(ds))

        # Stack bands
        nrg = np.dstack(img_bands)

        # View the color composite
        ax.imshow(nrg)
        ax.set_title(f'{title}')
        if show_plot:
            plt.show()
        return ax

    def plot_rbg(self, ax=None, show_plot=False, r=3, b=4, g=2):
        """
        Thank you aaron you're a G whoever you are!
        https://gis.stackexchange.com/questions/306164/how-to-visualize-multiband-imagery-using-rasterio

        :return:
        """
        if ax is None:
            fig, ax = plt.subplots()
        # Convert to numpy arrays
        blue = self.image.read(b)
        red = self.image.read(r)
        green = self.image.read(g)

        # Normalize band DN
        blue_norm = normalise(blue)
        red_norm = normalise(red)
        green_norm = normalise(green)

        # Stack bands
        nrg = np.dstack((blue_norm, red_norm, green_norm))

        # View the color composite
        ax.imshow(nrg)
        ax.set_title(f'{self.name}')
        if show_plot:
            plt.show()
        return ax

    def check_roi(self, roi):
        """
        Check that the ROI is within the bounds of the image otherwise make within bounds...
                    self.u.dp(['left edge coord:', self.image.bounds[0],
                      '\nbottom edge coord:', self.image.bounds[1],
                       '\nright edge coord:', self.image.bounds[2],
                       '\ntop edge coord:', self.image.bounds[3],
        :param roi:
        :return:
        """
        roi_n = {}
        roi_n['y1'] = roi['y1'] if roi['y1'] > self.image.bounds[1] else self.image.bounds[1]
        roi_n['y2'] = roi['y2'] if roi['y2'] < self.image.bounds[3] else self.image.bounds[3]
        roi_n['x1'] = roi['x1'] if roi['x1'] > self.image.bounds[0] else self.image.bounds[0]
        roi_n['x2'] = roi['x2'] if roi['x2'] < self.image.bounds[2] else self.image.bounds[2]
        for k, v in roi_n:
            if roi_n[k] != roi[k]:
                self.u.dp(['ROI was out of bounds, updated to be size of box...', k, roi_n[k], roi[k]])
        return roi_n

    def plot_subset(self, roi: dict, bands: list, ax=None, title='', show_plot=True):
        """
        Save a subset of an image based on a region of interest and the bands
        thayt you want to save.

        Eventually have this to save:
        https://rasterio.readthedocs.io/en/stable/quickstart.html#creating-data
        :param roi: region of interest e.g. build_polygon_from_centre_point or build_circle_from_centre_point
        :param bands: bands as a list
        :param ax: figure axes
        :param title:
        :param show_plot
        :return:
        """
        if ax is None:
            fig, ax = plt.subplots()
        # Convert to numpy arrays
        img_bands = []
        for b in bands:
            roi = self.check_roi(roi)
            ds = self.image.read(b)[roi['y1']:roi['y2'], roi['x1']:roi['x2']]  # Now filter if the pixels are in the ROI
            img_bands.append(normalise(ds))

        # Stack bands
        nrg = np.dstack(img_bands)

        # View the color composite
        ax.imshow(nrg)
        ax.set_title(f'{title}')
        if show_plot:
            plt.show()
        return ax

    def plot(self, band: int, ax=None, show_plot=True, downsample=None):
        """
        Plot specific bands.

        :param band:
        :param show_plot: whether or not to show th eplot
        :param ax:
        :return: ax
        """
        # Plot the first band
        band1 = self.image.read(band)
        return self.plot_idx(band1, ax, show_plot, downsample)

    def load_image(self, image_path=None, plot=False, normalise_bands=False):
        """
        https://rasterio.readthedocs.io/en/latest/quickstart.html
        :param image_path: Loads the ortho photo into memory
        :param plot: whether or not to plot it
        :return:
        """
        self.image = rasterio.open(image_path)
        if self._verbose:
            self.u.dp(['left edge coord:', self.image.bounds[0],
                      '\nbottom edge coord:', self.image.bounds[1],
                       '\nright edge coord:', self.image.bounds[2],
                       '\ntop edge coord:', self.image.bounds[3],
                       '\ndataset width:', self.image.width,
                       '\ndataset height:', self.image.height,
                       '\nnumber of bands:', self.image.indexes,  # Get bands
                       '\ngeo ref system:', self.image.crs,
                       '\ndata transform\n', self.image.transform
                       ])
        if normalise_bands:
            # Normalize band DN
            # # Write to TIFF
            new_dataset = rasterio.open(
                f'{image_path.replace(".tif", "_normalised.tif")}',
                'w',
                height=self.image.height,
                width=self.image.width,
                count=self.image.indexes,
                dtype=rasterio.float32,
                crs=self.image.crs,
                transform=self.image.transform,
            )
            # Normalise all the bands
            for band in range(1, self.image.indexes):
                new_dataset.write(normalise(self.image.read(band)), band)
            new_dataset.close()
            # Let the user know it has been written to a new file
            self.u.dp(['Wrote normalised tif to new file: ', f'{image_path.replace(".tif", "_normalised.tif")}'])
            self.image = rasterio.open(f'{image_path.replace(".tif", "_normalised.tif")}')

        if plot:
            self.plot(1)

    def mask_on_index(self, index, index_cutoff=0.5):
        """
        Mask the image based on the index passed (for example NDVI).
        Make a multiplier i.e. 0 and 1s where it is above the threshold otherwise have nothing.

        :param index:
        :param index_cutoff:
        :return:
        """
        # Convert pixels to 0 if we have a mask, not sure if this is the best way to do it...
        mask = index > index_cutoff
        return mask

    def write_as_rbg(self, filename: str, r=3, b=4, g=2):
        """
        Write the RGB version out to a tif file from a hyperspectral tif.
        The defaults correspond to bands from the planetscope data.

        :param r: Red band
        :param b: blue band
        :param g: green band
        :return:
        """
        blue = self.image.read(b)
        red = self.image.read(r)
        green = self.image.read(g)

        # Normalize band DN
        nir_norm = normalise(blue)
        red_norm = normalise(red)
        green_norm = normalise(green)

        # # Write to TIFF
        new_dataset = rasterio.open(
            filename,
            'w',
            height=nir_norm.shape[0],
            width=nir_norm.shape[1],
            count=3,
            dtype=rasterio.float32,
            crs=self.image.crs,
            transform=self.image.transform,
        )

        new_dataset.write(nir_norm, 1)
        new_dataset.write(red_norm, 2)
        new_dataset.write(green_norm, 3)
        new_dataset.close()


def get_values_for_location(image, lat, lon, bands_indices):
    # Gets all values, i.e. all bands, and all indicies
    y, x = image.image.index(lat, lon)
    # Now get all bands
    row = []
    for b in bands_indices:
        row.append(b[x, y])
    return row


def mask_values(index, lower_bound, upper_bound):
    """
    Mask the image based on the index passed (for example NDVI).
    Make a multiplier i.e. 0 and 1s where it is above the threshold otherwise have nothing.

    :param index:
    :param index_cutoff:
    :return:
    """
    # Convert pixels to 0 if we have a mask, not sure if this is the best way to do it...
    mask = np.ma.masked_less_equal(index, upper_bound)
    mask = np.ma.masked_greater_equal(mask.data*mask.mask, lower_bound)
    return mask.mask*1.0

