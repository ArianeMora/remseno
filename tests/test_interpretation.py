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


class TestInt(TestClass):

    def get_test_coords(self):
        c = Coords(drone_pine_coords, x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1='class1', class2='class2')
        c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32614", plot=True)
        return c

    def get_test_ortho(self):
        o = Image()
        o.load_image(image_path=drone_ortho)
        return o

    def test_sr(self):
        o = self.get_test_ortho()
        sr = o.get_sr(1, 2)
        o.plot_idx(sr)

    def test_mask_ndvi(self):
        o = Image()
        o.load_image(image_path='../data/public_data/waldi_july.tif')
        sr = o.get_sr(nir_band=8, red_band=6)
        sr = np.nan_to_num(sr)
        print(np.min(sr), np.max(sr))
        mask = o.mask_on_index(sr, 10)
        plt.imshow(mask)
        plt.show()
        # Check how it looks with the masking of the image
        plt.imshow(mask*o.get_band(1))
        plt.show()
