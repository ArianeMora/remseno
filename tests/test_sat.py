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

from remseno.indices import *
from tests.test_remseno import TestRemsenso

img_bands = [1, 2, 3, 4, 5, 6, 7, 8]

image = '/Users/ariane/Documents/code/remseno/scripts/4a2552c1-ea6d-4439-9da2-4d45ed54d049/PSScene/20220606_144123_29_2435_3B_AnalyticMS_SR_8b_clip.tif' #'../scripts/a69ebaa1-258f-4f50-8952-7344ac2f93bf/PSScene/20220606_144123_29_2435_3B_AnalyticMS_SR_8b_clip.tif' # '../data/tallo/planetscope/sat_data/dab9fa59-4a98-4319-ad5f-acea23bc6feb/PSScene/20230113_230924_81_241e_3B_AnalyticMS_SR_8b_clip.tif'
coords = '../data/tallo/planetscope/planetscope_test.csv'


class TestSat(TestRemsenso):
    def get_test_coords(self):
        c = Coords(coords, x_col='latitude', y_col='longitude', label_col='tree_id',
                   id_col='tree_id', sep=',', class1='T_498824', class2='T_498824', crs="EPSG:4326")
        c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32755", plot=True)
        return c

    def get_test_ortho(self):
        o = Image()
        o.load_image(image_path=image)
        return o

    def test_load_sat(self):
        # Test trianing a VAE for checking OOD
        o = self.get_test_ortho()
        c = self.get_test_coords()
        #c.plot_on_image(o, band=1)
        #plt.show()

    def test_ndvi(self):
        o = self.get_test_ortho()
        ndvi = get_ndvi(image=o.image, red_band=6, nir_band=8)
        o.plot_idx(ndvi)
        plt.show()

    def test_plot(self):
        o = self.get_test_ortho()
        o.plot_rbg()
        plt.show()

    def test_mask(self):
        o = self.get_test_ortho()
        ax = o.plot_rbg()
        ndvi = get_ndvi(image=o.image, red_band=6, nir_band=8)
        mask = o.mask_on_index(ndvi, 0.75)
        plt.imshow(mask*ndvi)
        plt.show()

    def test_write_band(self):
        o = self.get_test_ortho()
        o.write_as_rbg('output.tif')
