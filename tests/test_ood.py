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


drone_dir = '../data/dryad_trees/'
drone_ortho = '../data/dryad_trees/Stitch_Image/20190518_pasture_100ft_RGB_GCPs_Forest.tif'
drone_coords = '../data/dryad_trees/dryad_cedar_pine/theredcedar_xy.csv'
drone_pine_coords = '../data/dryad_trees/dryad_cedar_pine/pine_class.csv'


class TestOOD(TestRemsenso):

    def get_test_coords(self):
        c = Coords(drone_pine_coords, x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1='class1', class2='class2')
        c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32614", plot=True)
        return c

    def get_test_ortho(self):
        o = Image()
        o.load_image(image_path=drone_ortho)
        return o

    def test_image(self):
        # Test image loading
        o = self.get_test_ortho()
        o.plot(1)
        o.plot(2)


    def test_ood_train(self):
        # Test trianing a VAE for checking OOD
        o = Image()
        o.load_image(image_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ood = OOD(o, c)
        dist = ood.train_ood(image=o, coords=c, bands=[o.get_band(b) for b in [1, 2, 3, 4]],
                                 max_pixel_padding=3)
        plt.hist(dist[:, 0])
        plt.show()

    def test_load_ood(self):
        # Load presaved model
        o = Image()
        o.load_image(image_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ood = OOD(o, c)
        ood.load_saved_vae()
        print('loaded')

    def test_ood_predict(self):
        # Test trianing a VAE for checking OOD
        o = Image()
        o.load_image(image_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ood = OOD(o, c)
        ood.load_saved_vae()
        print('loaded')
        # Now check prediction using waeldi ood

        oo_c = Coords('data/ood_waeldi.csv', x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1=0, class2=0)
        # Transform coords
        oo_c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:21781", plot=False) #EPSG: 4326

        # Build train df --> needs to be as above
        labels = [f'b{i}' for i in [1, 2, 3, 4]]
        df, labels = ood.build_train_df(o, oo_c, bands=[o.get_band(b) for b in [1, 2, 3, 4]],
                                 max_pixel_padding=1, band_labels=labels)

        df['pred'], pred = ood.classify_ood(df[labels].values)
        print(df)
        plt.hist(pred[:, 0])
        plt.show()
        print(len(df[df['pred'] == True]))
        df, labels = ood.build_train_df(o, c, bands=[o.get_band(b) for b in [1, 2, 3, 4]],
                                        max_pixel_padding=1, band_labels=labels)
        df['pred'], pred = ood.classify_ood(df[labels].values)
        print(df)
        plt.hist(pred[:, 0])
        print(len(df[df['pred'] == True]))
        plt.show()


    def test_ood_train_hyper(self):
        # Test trianing a VAE for checking OOD
        o = Image()
        o.load_image(image_path='../data/public_data/waldi_july.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        c.transform_coords(tree_coords="EPSG:21781", image_coords="EPSG:4326", plot=False) #EPSG: 4326

        oo_c = Coords('data/ood_waeldi.csv', x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1=0, class2=0)
        ood = OOD(o, c)
        ood.load_saved_vae()
        #
        # dist = ood.train_ood(image=o, coords=c, bands=[o.get_band(b) for b in [1, 2, 3, 4, 5, 6, 7, 8]],
        #                      max_pixel_padding=3)
        plt.show()
        bands = [1, 2, 3, 4, 5, 6, 7, 8]
        labels = [f'b{i}' for i in bands]
        df, labels = ood.build_train_df(o, oo_c, bands=[o.get_band(b) for b in bands],
                                 max_pixel_padding=1, band_labels=labels)

        df['pred'], pred = ood.classify_ood(df[labels].values)
        print(df)
        plt.hist(pred[:, 0])
        plt.show()
        print(len(df[df['pred'] == True]))
        df, labels = ood.build_train_df(o, c, bands=[o.get_band(b) for b in bands],
                                        max_pixel_padding=1, band_labels=labels)
        df['pred'], pred = ood.classify_ood(df[labels].values)
        print(df)
        plt.hist(pred[:, 0])
        print(len(df[df['pred'] == True]))
        plt.show()