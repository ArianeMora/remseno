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

import shutil
import tempfile
import unittest

from remseno.indices import *
from tests.test_remseno import TestRemsenso


drone_dir = '../data/dryad_trees/'
drone_ortho = '../data/dryad_trees/Stitch_Image/20190518_pasture_100ft_RGB_GCPs_Forest.tif'
drone_coords = '../data/dryad_trees/dryad_cedar_pine/theredcedar_xy.csv'
drone_pine_coords = '../data/dryad_trees/dryad_cedar_pine/pine_class.csv'


class TestML(TestRemsenso):

    def test_ml(self):
        ml = ML()
        o = self.get_test_ortho()
        c = self.get_test_coords()
        ml.binary_classifier(image=o, coords=c, bands=[1, 2, 3])

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
        o = self.get_test_ortho()
        c = self.get_test_coords()
        ood = OOD(o, c)
        tdf = ood.build_train_df(image=o, coords=c, bands=[o.get_band(b) for b in [1, 2, 3]],
                                 max_pixel_padding=3)
        print(tdf)