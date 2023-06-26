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
Here we just want to load the file with the trees in it. This should be in csv format.

User input is required here! You just need to provide information from your file.

Inputs:

1. path to the file with the label
2. x coord column name
3. y cood column name
4. label = the column name with the different labels (expecting a 2 class classifier)
5. name id name for your file
6. separator (i.e. comma, t, or ;)
7. class1 = value of class1 i.e. Sylvatica
8. class2 = value of class2 i.e. Orientalis
"""

from remseno.base import Remsenso
import asyncio
import os

import planet
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer
import rasterio.warp
from rasterio.crs import CRS
import rasterio

class Coords(Remsenso):

    def __init__(self, file_path: str, x_col: str, y_col: str,
                 label_col: str, id_col: str, class1, class2, crs,
                 binary_label='binary_label', sep=','):
        super().__init__()
        self.df = None
        self.x_col, self.y_col, self.label_col, self.id_col, self.class1, self.class2, \
        self.sep = x_col, y_col, label_col, id_col, class1, class2, sep
        self.binary_label = binary_label
        self.crs = crs # Coordinate reference system
        self.load(file_path)

    def load(self, file_path, plot=False):
        df = pd.read_csv(file_path, sep=self.sep)
        # Filter the dataframe to only include the two classes
        original_size = len(df)
        #df = df[df[self.label_col].isin([self.class1, self.class2])]
        self.u.dp(['Removed rows not in class1 or class2 i.e.', self.class1, self.class2, ' in column:', self.label_col,
                   'Your dataset origionally had:', original_size, '\nNow you have:', len(df)])

        # Make the classes based on the labels provided
        df[self.binary_label] = [0 if c == self.class1 else 1 for c in df[self.label_col].values]
        self.df = df
        if plot:
            # Plot the X and Y coordinates
            plt.scatter(df[self.x_col], df[self.y_col], c=df[self.binary_label].values)
            # Drop the outlier and looks about right woo!
            plt.title('Coords scatter')
            plt.xlabel(self.x_col)
            plt.ylabel(self.y_col)
            plt.show()

        # If you notice that it looks wrong you might need to remove some data or points
    def plot_on_image(self, image, band=1):
        """
        Plot the coords on an orthosomaic
        :param ortho:
        :return:
        """
        fig, ax = plt.subplots()
        ax = image.plot(band=band, ax=ax, show_plot=False)
        ys = self.df[self.y_col].values
        self.df[f'colour'] = ['blue' if c == 0 else 'red' for c in self.df['binary_label'].values]
        colours = self.df[f'colour'].values

        for i, x in enumerate(self.df[self.x_col].values):
            y, x = image.image.index(x, ys[i])
            ax.scatter(x, y, c=colours[i])
        return ax

    def transform_coord(self, x: float, y: float, src: str, dest: str):
        """
        transform
        :param x:
        :param y:
        :param src:
        :param dest:
        :return:
        """
        transformer = Transformer.from_crs(src, dest)
        return transformer.transform(x, y)  # Double check this on QGIS

    def transform_coords(self, tree_coords: str, image_coords: str, plot=True):
        """
        ## 5. Transform coordinates if necessary

        The geo ref system print out above tells you what the coordinate system is, if that is different to the one
        in your tree file you will need to update these as necessary. I chose to update the coords of the trees.

        You need to know the source and destination coord system
        https://docs.opendronemap.org/arguments/
        For this I use pyproj:
        https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1

        Documentation: https://pyproj4.github.io/pyproj/stable/

        :param tree_coords: e.g. "EPSG:21781"
        :param ortho_coords: e.g. "EPSG:3044" (google maps 4326)
        :param plot
        :return:
        """

        transformer = Transformer.from_crs(tree_coords, image_coords)

        # Now we transform each of the x and y params
        # Convert x and y to the new coord system
        converted_x = []
        converted_y = []
        original_y = self.df[self.y_col].values
        for i, x in enumerate(self.df[self.x_col]):
            x_wsg, y_wsg = transformer.transform(x, original_y[i])  # Double check this on QGIS
            converted_x.append(x_wsg)
            converted_y.append(y_wsg)

        # Save the old coords
        self.df[f'{self.x_col}_{tree_coords}'] = self.df[self.x_col].values
        self.df[f'{self.y_col}_{tree_coords}'] = self.df[self.y_col].values

        # Now add in the new coords
        self.df[self.x_col] = converted_x
        self.df[self.y_col] = converted_y

        if plot:
            # Replot them now
            plt.scatter(self.df[self.x_col], self.df[self.y_col], c=self.df[self.binary_label].values)
            # Drop the outlier and looks about right woo!
            plt.title(f'Coords transformed: {tree_coords} --> {image_coords}')
            plt.xlabel(self.x_col)
            plt.ylabel(self.y_col)
            plt.show()

    def build_circle_from_centre_point(self, x: int, y: int, radius_pixels: int):
        """
        Could be useful for doing something like this:
        https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_radial_mean.html#sphx-glr-advanced-image-processing-auto-examples-plot-radial-mean-py
        :param x: index of picture we want to look at (i.e. pixel x)
        :param y: index of picture we want to look at (i.e. pixel y)
        :param radius_pixels: radius in pixels
        :return:
        """
        circle_xys = []
        r2 = radius_pixels**2  # Radius squared
        xmin = x - radius_pixels
        xmax = x + radius_pixels
        ymin = y - radius_pixels
        ymax = y + radius_pixels
        #ToDo check for size good tech debt
        for xi in range(xmin, xmax):  # Should really check the size but YOLO
            for yi in range(ymin, ymax):
                dx = xi - x
                dy = yi - y
                dists = (dx**2) + (dy**2)
                if dists < r2:
                    circle_xys.append([xi, yi])

        return circle_xys

    def build_polygon_from_centre_point(self, lat: float, lon: float, width_m: float, height_m: float, crs: str):
        """
        Build a polygon shape for using to download planet scope or other satelite data.
        :param lat:
        :param long:
        :param width_m:
        :param height_m:
        :return: polygon/square...
        """
        # First convert to EPSG if not in the right one
        if crs != 'EPSG:4326':
            lat, lon = self.transform_coord(lat, lon, crs, 'EPSG:4326')
        # Get the width and offset each of the points
        w = width_m/2
        h = height_m/2
        poly = [self.offset(lat, lon, -1 * w, h),  # LH corner
                self.offset(lat, lon, -1 * w, -1 * h),  # UP
                self.offset(lat, lon, w, -1 * h),  # Right
                self.offset(lat, lon, w, h),  # Right corner
                self.offset(lat, lon, -1 * w, h)]  # Start again
        # Convert back
        if crs != 'EPSG:4326':
            # Project the feature to the desired CRS
            coors = [self.transform_coord(b[0], b[1], 'EPSG:4326', crs) for b in poly]
            return coors

        return poly

    def offset_latitude(self, lat: float, distance_x_m: float):
        """
        Offset latitude.
        :param lat:
        :param distance_x_m:
        :return:
        """

        # Earth’s radius, sphere
        R = 6378137

        # Coordinate offsets in radians
        dLat = distance_x_m/111000 # distance_x_m / R
        return lat + dLat #* 180 / np.pi

    def offset_longditude(self, lon: float, distance_y_m: float):
        """
        Offset longditude.

        Thank you Hugo Palmer m8y
        https://gis.stackexchange.com/questions/289111/how-to-add-meters-to-epsg4326-coordinates
        :param lon:
        :param distance_y_m:
        :return:
        """
        # Earth’s radius, sphere
        R = 6378137

        # Coordinate offsets in radians
        dLon = distance_y_m / 80000  # (R * np.cos(np.pi * lon / 180))

        # OffsetPosition, decimal degrees
        return lon + dLon #* 180 / np.pi

    def offset(self, lat: float, lon: float, distance_x_m: float, distance_y_m: float):
        """
        Offset a coord file (x, y) by certain number of m's

        Thanks so ol m8 haakon_d from stack overflow:
        https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters

        :param distance_x_m: distance to offset in m
        :param distance_y_m: distance to offset in m
        :param lon: longditude
        :param lat: latitude
        :return:
        """
        # Earth’s radius, sphere
        return self.offset_latitude(lat, distance_x_m), self.offset_longditude(lon, distance_y_m)

    async def download_planetscope(self, download_dir, df, planet_api_key: str, lat: str, lon: str, image_id: str,
                                   distance_x_m: float, distance_y_m: float):
        os.environ['PL_API_KEY'] = 'PLAK5a21e86c2faf452195d43c3ca3f318ee'
        async with planet.Session() as sess:
            client = sess.client('orders')
            lats = df[lat].values
            lons = df[lon].values
            for i, img_id in enumerate(df[image_id].values):
                requests = [self.create_requests(img_id, lats[i], lons[i], distance_x_m, distance_y_m)]

            await asyncio.gather(*[
                create_and_download(client, request, download_dir)
                for request in requests
            ])

    def create_requests(self, image_id: str, lat: float, lon: float, distance_x_m: float, distance_y_m: float):
            """
            Downloads planet scope data
            :param image_id: id of the image
            :param lat:
            :param lon:
            :param distance_x_m:
            :param distance_y_m:
            :return:
            """
            aoi = self.build_polygon_from_centre_point(lat, lon, distance_x_m, distance_y_m, self.crs)
            dl_aoi = {
                "type":
                    "Polygon",
                "coordinates": [aoi]
            }

            dl_items = [image_id]
            dl_order = planet.order_request.build_request(
                name='iowa_order',
                products=[
                    planet.order_request.product(item_ids=dl_items,
                                                 product_bundle='analytic_8b_sr_udm2',
                                                 item_type='PSScene')
                ],
                tools=[planet.order_request.clip_tool(aoi=dl_aoi)])
            return dl_order


async def create_and_download(client, order_detail, directory):
    """Make an order, wait for completion, download files as a single task."""
    with planet.reporting.StateBar(state='creating') as reporter:
        order = await client.create_order(order_detail)
        reporter.update(state='created', order_id=order['id'])
        await client.wait(order['id'], callback=reporter.update_state)

    await client.download_order(order['id'], directory, progress_bar=True)
