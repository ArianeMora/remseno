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
drone_coords = '../data/dryad_trees/dryad_cedar_pine/theredcedar_xy.csv'
drone_pine_coords = '../data/dryad_trees/dryad_cedar_pine/pine_class.csv'


# df = pd.read_csv(drone_pine_coords)
# df['class'] = ['class1' if i % 2 == 0 else 'class2' for i in range(0, len(df))]
# df.to_csv('../data/dryad_trees/dryad_cedar_pine/pine_class.csv', index=False)


class TestML(TestClass):

    def get_test_coords(self):
        c = Coords(drone_pine_coords, x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1='class1', class2='class2')
        c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32614", plot=True)
        return c

    def get_test_ortho(self):
        o = Image()
        o.load_image(image_path=drone_ortho)
        return o


    def test_ml(self):
        ml = ML()
        o = Image()
        o.load_image(image_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ml.binary_classifier(image=o, coords=c, bands=[1, 2, 3, 4])

    def test_training_ml(self):
        # Make a list of training datasets
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        o1 = Image()
        o1.load_image(image_path='../data/public_data/waeldi.tif')

        o2 = Image()
        o2.load_image(image_path='../data/public_data/waeldi.tif')

        o3 = Image()
        o3.load_image(image_path='../data/public_data/waeldi.tif')

        ml = ML()
        train_df = ml.create_training_dataset(image_list=[o1, o2, o3], bands=[1, 2], coords=c, max_pixel_padding=2)
        print(train_df.head())

    def test_train_df(self):
        o = Image()
        o.load_image(image_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ood = OOD()
        tdf = ood.build_train_df(image=o, coords=c, bands=[o.get_band(b) for b in [1, 2, 3, 4]],
                                 max_pixel_padding=3)
        print(tdf)