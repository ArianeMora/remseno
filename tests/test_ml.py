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


class TestML(TestRemsenso):

    def test_ml(self):
        ml = ML()
        o = self.get_test_ortho()
        c = self.get_test_coords()
        ml.binary_classifier(image=o, coords=c, bands=[1, 2, 3])

    def test_training_ml(self):
        # Make a list of training datasets
        c = self.get_test_coords()
        o1 = self.get_test_ortho()
        ml = ML()
        df = ml.train_ml(o1, bands=[1, 2, 3], coords=c, max_pixel_padding=2)
        print(df)

    def test_train_df(self):
        o = self.get_test_ortho()
        c = self.get_test_coords()
        ood = OOD(o, c)
        tdf = ood.build_train_df(image=o, coords=c, bands=[o.get_band(b) for b in [1, 2, 3]],
                                 downsample=3)
        print(tdf)