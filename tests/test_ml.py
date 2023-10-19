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
        bands.append(get_ndvi(o1.image, 1, 2))
        df = ml.train_ml(clf, image=o1, coords=c, image_bands=bands, validation_percent=20, test_percent=20,
                         max_pixel_padding=2, normalise=False)
        df.to_csv('data/test_pred.csv')

    def test_training_ml_multiple_images(self):
        # Make a list of training datasets
        c = self.get_test_coords()
        o1 = self.get_test_ortho()
        o2 = self.get_test_ortho()
        ml = ML()
        # Make a classifier any from sklearn
        clf = svm.SVC(C=8.0, kernel='poly', class_weight='balanced')

        images = [{'image': o1,
                              'indexs': {'ndvi': get_ndvi(o1.image, 1, 2),
                                         'sr': get_sr(o1.image, nir_band=1, red_band=3),
                                         'band_2': o1.image.read(2),
                                         'band_3': o1.image.read(3),
                                         }},

                  {'image': o2, # The image read in
                  'indexs': {'gndvi': get_gndvi(o1.image, 1, 2), # Any of the indicies or you could just do a band
                             'get_normg': get_normg(o1.image, red_band=1, blue_band=2, green_band=3),
                             'ndvi': get_ndvi(o1.image, 1, 2),
                             'band_1': o1.image.read(1),
                             'band_2': o1.image.read(2),
                             'band_3': o1.image.read(3),
                             }},
                ]
        df = ml.train_ml_on_multiple_images(clf, images=images, coords=c, validation_percent=20, test_percent=20,
                                            max_pixel_padding=2, normalise=False)
        df.to_csv('data/test_pred_multiple_images.csv')

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
        df.to_csv('data/test_pred.csv')



