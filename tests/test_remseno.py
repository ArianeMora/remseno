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

import os
import shutil
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from remseno import *
from remseno.indices import *


class TestClass(unittest.TestCase):

    @classmethod
    def setup_class(self):
        local = True
        # Create a base object since it will be the same for all the tests
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))

        self.data_dir = os.path.join(THIS_DIR, 'remsenso/')
        if local:
            self.tmp_dir = os.path.join(THIS_DIR, 'remsenso/tmp/')
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.mkdir(self.tmp_dir)
        else:
            self.tmp_dir = tempfile.mkdtemp(prefix='remsenso')

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)


drone_dir = '../data/dryad_trees/'
drone_ortho = '../data/dryad_trees/Stitch_Image/20190518_pasture_100ft_RGB_GCPs_Forest.tif'
drone_coords = '../data/dryad_trees/location_files/Annotations.csv'
drone_pine_coords = '../data/dryad_trees/location_files/Annotations.csv' #'../data/dryad_trees/dryad_cedar_pine/pine_class.csv'


# df = pd.read_csv(drone_pine_coords)
# df['class'] = ['class1' if i % 2 == 0 else 'class2' for i in range(0, len(df))]
# df.to_csv('../data/dryad_trees/dryad_cedar_pine/pine_class.csv', index=False)


class TestRemsenso(TestClass):

    def get_test_coords(self):
        c = Coords(drone_pine_coords, x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1='RedCedar', class2='Pine', crs="EPSG:4326")
        c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32614", plot=True)
        return c

    def get_test_ortho(self):
        o = Image()
        o.load_image(image_path=drone_ortho)
        return o

    def test_get_units(self):
        o = self.get_test_ortho()
        o.get_pix_to_m()

    def test_plot_downsample(self):
        o = self.get_test_ortho()
        o.plot(1, downsample=10)
        plt.show()

    def test_image(self):
        # Test image loading
        o = self.get_test_ortho()
        o.plot(1)
        o.plot(2)

    def test_coords(self):
        # Test coords loading: Tech debt need to swap X and Y over...
        c = self.get_test_coords()
        o = self.get_test_ortho()
        c.plot_on_image(o)
        plt.show()

    def test_draw_bb(self):
        # Test coords loading
        c = self.get_test_coords()
        df = c.df
        x = df[c.x_col].values[0]
        y = df[c.y_col].values[0]
        bb = c.build_polygon_from_centre_point(x, y, 2, 2, c.crs)
        print(bb)
        xs = [b[0] for b in bb]
        ys = [b[1] for b in bb]
        for b in bb:
            print(b)
        plt.plot(xs, ys)
        plt.title("Bounding box plot")
        plt.show()

    def test_multi_band(self):
        o = self.get_test_ortho()
        o.plot_multi_bands([1, 2, 3])
        plt.show()

    def test_multi_band_subset(self):
        o = self.get_test_ortho()
        c = self.get_test_coords()
        df = c.df
        fig, ax = plt.subplots()
        x = df[c.x_col].values[0]
        y = df[c.y_col].values[0]
        y, x = o.image.index(x, y)
        bb = c.build_circle_from_centre_point(x, y, 3)
        xs = []
        ys = []
        ax = o.plot_multi_bands([1, 2, 3])
        for b in bb:
            xs.append(b[0])
            ys.append(b[1])
            ax.scatter(b[0], b[1], s=3)
        plt.title("Bounding box circle")

        # Add this to get sub-image
        pixel_buffer = 20
        plt.xlim([min(xs) - pixel_buffer, max(xs) + pixel_buffer])
        plt.ylim([min(ys) - pixel_buffer, max(ys) + pixel_buffer])
        plt.show()

    def test_plot_circle(self):
        # Test coords loading
        o = self.get_test_ortho()
        c = self.get_test_coords()
        df = c.df
        ax = o.plot(2, show_plot=False)
        x = df[c.x_col].values[0]
        y = df[c.y_col].values[0]
        y, x = o.image.index(x, y)
        bb = c.build_circle_from_centre_point(x, y, 100)
        xs = []
        ys = []
        for b in bb:
            xs.append(b[0])
            ys.append(b[1])
        ax.scatter(xs, ys, s=8)
        plt.title("Bounding box circle")
        pixel_buffer = 1000
        plt.xlim([min(xs) - pixel_buffer, max(xs) + pixel_buffer])
        plt.ylim([min(ys) - pixel_buffer, max(ys) + pixel_buffer])
        plt.show()

    def test_plot_subset_circle(self):
        # Test plotting just a subset in mutliple bands --> good for large images!
        o = self.get_test_ortho()
        c = self.get_test_coords()
        df = c.df
        x = df[c.x_col].values[0]
        y = df[c.y_col].values[0]
        y, x = o.image.index(x, y)
        bb = c.build_circle_from_centre_point(x, y, 20)
        xs = []
        ys = []
        for b in bb:
            xs.append(b[0])
            ys.append(b[1])
        plt.title("Bounding box circle")
        pixel_buffer = 1000
        roi = {'x1': min(xs) - pixel_buffer, 'x2': max(xs) + pixel_buffer,
               'y1': min(ys) - pixel_buffer, 'y2': max(ys) + pixel_buffer}
        ax = o.plot_subset(roi, [1, 2, 3], show_plot=False)
        xs = [pixel_buffer + (x - min(xs)) for x in xs]
        ys = [pixel_buffer + (y - min(ys)) for y in ys]
        ax.scatter(xs, ys, s=8)
        plt.show()

    def test_draw_circle(self):
        # Test coords loading
        o = self.get_test_ortho()
        c = self.get_test_coords()
        df = c.df
        x = df[c.x_col].values[0]
        y = df[c.y_col].values[0]
        y, x = o.image.index(x, y)
        bb = c.build_circle_from_centre_point(x, y, 2)
        for b in bb:
            plt.scatter(b[0], b[1])
        plt.title("Bounding box circle")
        plt.show()

    def test_circle_on_image(self):
        # Test coords loading
        o = self.get_test_ortho()
        c = self.get_test_coords()
        df = c.df
        ax = o.plot(2, show_plot=False)
        for i in range(0, 2):
            x = df[c.x_col].values[i]
            y = df[c.y_col].values[i]
            y, x = o.image.index(x, y)
            bb = c.build_circle_from_centre_point(x, y, 8)
            for b in bb:
                ax.scatter(b[0], b[1], s=2)
        plt.title("Bounding box circle")
        plt.show()

    def test_draw_bb_on_image(self):
        o = self.get_test_ortho()
        c = self.get_test_coords()
        ax = c.plot_on_image(o)
        # Coords need to be transformed... then transformed back
        #c.transform_coords("EPSG:32614", "EPSG:4326", plot=False)
        df = c.df

        for i in range(0, len(df)):
            x = df[c.x_col].values[i]
            y = df[c.y_col].values[i]
            bb = c.build_polygon_from_centre_point(x, y, 2, 2, "EPSG:32614")
            bb = [o.image.index(b[0], b[1]) for b in bb]
            xs = [b[1] for b in bb]
            ys = [b[0] for b in bb]
            ax.plot(xs, ys)
        plt.title("Bounding box plot")
        plt.show()

    def test_download(self):
        df = pd.read_csv(f'../data/output/planetscope/download_DF_dedup.csv')
        #print(df.head())
        #df = df.head(4)
        c = self.get_test_coords()
        data = []
        image_ids = df['image_ids'].values
        lats = df['latitude'].values
        longs = df['longitude'].values
        for i in range(10, 20):
            aoi = c.build_polygon_from_centre_point(lats[i], longs[i], 500, 500, "EPSG:4326")
            # For some reason need to swap it around classic no idea why...
            aoi = [[p[1], p[0]] for p in aoi]
            data.append([aoi, image_ids[i]])

        asyncio.run(download(data))
        # c = self.get_test_coords()
        # asyncio.run(c.download_planetscope('/Users/ariane/Documents/code/remseno/data/output/planetscope/images/', df,
        #                        '', 'latitude', 'longitude', 'image_ids', distance_x_m=90, distance_y_m=90))