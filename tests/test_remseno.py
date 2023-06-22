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
from remseno.indices import Index


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


class TestRemsenso(TestClass):

    def test_ortho(self):
        # Test ortho loading
        o = Ortho()
        #o.load_ortho(ortho_path='../data/public_data/ortho_waeldi_agisoft_2.tif') #waeldi.tif')
        #o.load_ortho(ortho_path='../scripts/172110d8-234b-422a-b2bc-137d3792e666/PSScene/20220302_085116_40_241b_3B_AnalyticMS_SR_8b_clip.tif')
        o.load_ortho(ortho_path='/Users/ariane/Desktop/Raw_Images/DJI_0216.png')#Stitch_Image/20190518_pasture_100ft_RGB_GCPs_Forest.tif')
        o.plot(1)
        o.plot(2)

    def test_draw_bb(self):
        # Test coords loading
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        c.transform_coords(tree_coords="EPSG:21781", ortho_coords="EPSG:4326", plot=False) #EPSG: 4326
        df = c.df
        x = df[c.x_col].values[0]
        y = df[c.y_col].values[0]
        bb = c.build_polygon_from_centre_point(x, y, 20, 20)
        xs = [b[0] for b in bb]
        ys = [b[1] for b in bb]
        for b in bb:
            print(b)
        plt.plot(xs, ys)
        plt.title("Bounding box plot")
        plt.show()

    def test_multi_band(self):
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waldi_april.tif')
        o.plot_multi_bands()
        plt.show()

    def test_multi_band_subset(self):
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waldi_april.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        c.transform_coords(tree_coords="EPSG:21781", ortho_coords="EPSG:32632", plot=False) #EPSG: 4326
        df = c.df
        fig, ax = plt.subplots()
        x = df[c.x_col].values[0]
        y = df[c.y_col].values[0]
        y, x = o.ortho.index(x, y)
        bb = c.build_circle_from_centre_point(x, y, 3)
        xs = []
        ys = []
        ax = o.plot_multi_bands(ax)
        for b in bb:
            xs.append(b[0])
            ys.append(b[1])
            ax.scatter(b[0], b[1], s=3)
        plt.title("Bounding box circle")

        # Add this to get subimage
        pixel_buffer = 20
        plt.xlim([min(xs) - pixel_buffer, max(xs) + pixel_buffer])
        plt.ylim([min(ys) - pixel_buffer, max(ys) + pixel_buffer])
        plt.show()

    def test_plot_circle(self):
        # Test coords loading
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        df = c.df
        ax = o.plot(2, show_plot=False)

        x = df[c.x_col].values[0]
        y = df[c.y_col].values[0]
        y, x = o.ortho.index(x, y)
        bb = c.build_circle_from_centre_point(x, y, 8)
        xs = []
        ys = []
        for b in bb:
            xs.append(b[0])
            ys.append(b[1])
            ax.scatter(b[0], b[1], s=8)
        plt.title("Bounding box circle")
        pixel_buffer = 10
        plt.xlim([min(xs) - pixel_buffer, max(xs) + pixel_buffer])
        plt.ylim([min(ys) - pixel_buffer, max(ys) + pixel_buffer])
        plt.show()


    def test_draw_circle(self):
        # Test coords loading
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        c.transform_coords(tree_coords="EPSG:21781", ortho_coords="EPSG:4326", plot=False) #EPSG: 4326
        df = c.df
        x = df[c.x_col].values[0]
        y = df[c.y_col].values[0]
        y, x = o.ortho.index(x, y)
        bb = c.build_circle_from_centre_point(x, y, 5)
        for b in bb:
            plt.scatter(b[0], b[1])
        plt.title("Bounding box circle")
        plt.show()

    def test_circle_on_ortho(self):
        # Test coords loading
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        df = c.df
        ax = o.plot(2, show_plot=False)
        for i in range(0, len(df)):
            x = df[c.x_col].values[i]
            y = df[c.y_col].values[i]
            y, x = o.ortho.index(x, y)
            bb = c.build_circle_from_centre_point(x, y, 8)
            for b in bb:
                ax.scatter(b[0], b[1], s=2)
        plt.title("Bounding box circle")
        plt.show()

    def test_draw_bb_on_ortho(self):
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ax = c.plot_on_ortho(o)
        # Transform to make bb
        c.transform_coords(tree_coords="EPSG:21781", ortho_coords="EPSG:4326", plot=False) #EPSG: 4326
        df = c.df
        for i in range(0, len(df)):
            x = df[c.x_col].values[i]
            y = df[c.y_col].values[i]
            bb = c.build_polygon_from_centre_point(x, y, 20, 20)
            bb = [c.transform_coord(b[0], b[1], "EPSG:4326", "EPSG:21781") for b in bb]
            bb = [o.ortho.index(b[0], b[1]) for b in bb]
            xs = [b[1] for b in bb]
            ys = [b[0] for b in bb]
            for b in bb:
                print(b)

            ax.plot(xs, ys)
        plt.title("Bounding box plot")
        plt.show()

    def test_coords(self):
        # Test coords loading
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c.plot_on_ortho(o)

    def test_sr(self):
        img = Index()
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waldi_july.tif')
        sr = img.get_sr(image= o.ortho, nir_band=8, red_band=6, )
        o.plot_idx(sr)

    def test_ml(self):
        ml = ML()
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ml.binary_classifier(ortho=o, coords=c, bands=[1, 2, 3, 4])

    def test_training_ml(self):
        # Make a list of training datasets
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        o1 = Ortho()
        o1.load_ortho(ortho_path='../data/public_data/waeldi.tif')

        o2 = Ortho()
        o2.load_ortho(ortho_path='../data/public_data/waeldi.tif')

        o3 = Ortho()
        o3.load_ortho(ortho_path='../data/public_data/waeldi.tif')

        ml = ML()
        train_df = ml.create_training_dataset(image_list=[o1, o2, o3], bands=[1, 2], coords=c, max_pixel_padding=2)
        print(train_df.head())

    def test_train_df(self):
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ood = OOD()
        tdf = ood.build_train_df(image=o, coords=c, bands=[o.get_band(b) for b in [1, 2, 3, 4]],
                                 max_pixel_padding=3)
        print(tdf)

    def test_ood_train(self):
        # Test trianing a VAE for checking OOD
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ood = OOD(o, c)
        dist = ood.train_ood(image=o, coords=c, bands=[o.get_band(b) for b in [1, 2, 3, 4]],
                                 max_pixel_padding=3)
        plt.hist(dist[:, 0])
        plt.show()

    def test_load_ood(self):
        # Load presaved model
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ood = OOD(o, c)
        ood.load_saved_vae()
        print('loaded')

    def test_ood_predict(self):
        # Test trianing a VAE for checking OOD
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waeldi.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        ood = OOD(o, c)
        ood.load_saved_vae()
        print('loaded')
        # Now check prediction using waeldi ood

        oo_c = Coords('data/ood_waeldi.csv', x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1=0, class2=0)
        # Transform coords
        oo_c.transform_coords(tree_coords="EPSG:4326", ortho_coords="EPSG:21781", plot=False) #EPSG: 4326

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

    def test_mask_ndvi(self):
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waldi_july.tif')
        sr = o.get_sr(nir_band=8, red_band=6)
        sr = np.nan_to_num(sr)
        print(np.min(sr), np.max(sr))
        mask = o.mask_on_index(sr, 10)
        plt.imshow(mask)
        plt.show()
        # Check how it looks with the masking of the image
        plt.imshow(mask*o.get_band(1))
        plt.show()

    def test_ood_train_hyper(self):
        # Test trianing a VAE for checking OOD
        o = Ortho()
        o.load_ortho(ortho_path='../data/public_data/waldi_july.tif')
        c = Coords('../data/public_data/Waeldi_Adults_genotyped.csv', x_col='X', y_col='Y', label_col='Taxa',
                   id_col='ProbeIDoriginal', sep=',', class1='Sylvatica', class2='Orientalis')
        c.transform_coords(tree_coords="EPSG:21781", ortho_coords="EPSG:4326", plot=False) #EPSG: 4326

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