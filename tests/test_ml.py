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
from sklearn.tree import DecisionTreeClassifier


class TestML(TestRemsenso):

    def test_training_ml(self):
        # Make a list of training datasets
        c = self.get_test_coords()
        o1 = self.get_test_ortho()
        ml = ML()
        # Make a classifier any from sklearn
        clf = svm.SVC(C=8.0, kernel='poly', class_weight='balanced')
        bands = [o1.image.read(b) for b in [1, 2, 3]]
        # Also add an index
        bands.append(get_ndvi(o1, 1, 2))
        df = ml.train_ml(clf, image=o1, coords=c, image_bands=bands, validation_percent=20, test_percent=20,
                 max_pixel_padding=2, normalise=False)
        df.to_csv('test_pred.csv')

    def test_different_training_ml(self):
        # Make a list of training datasets
        c = self.get_test_coords()
        o1 = self.get_test_ortho()
        ml = ML()
        # Make a classifier any from sklearn
        clf = DecisionTreeClassifier(random_state=0)
        bands = [o1.image.read(b) for b in [1, 2, 3]]
        df = ml.train_ml(clf, image=o1, coords=c, image_bands=bands, validation_percent=20, test_percent=20,
                         max_pixel_padding=2, normalise=False)
        df.to_csv('test_pred.csv')



