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


class TestInt(TestRemsenso):

    def test_sr(self):
        o = self.get_test_ortho()
        sr = o.get_sr(1, 2)
        o.plot_idx(sr)
        plt.show()

    def test_mask_ndvi(self):
        o = self.get_test_ortho()
        sr = o.get_sr(nir_band=8, red_band=6)
        sr = np.nan_to_num(sr)
        print(np.min(sr), np.max(sr))
        mask = o.mask_on_index(sr, 10)
        plt.imshow(mask)
        plt.show()
        # Check how it looks with the masking of the image
        plt.imshow(mask*o.get_band(1))
        plt.show()
