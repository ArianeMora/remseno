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

img_bands = [1, 2, 3]


class TestOOD(TestRemsenso):

    def test_ood_train(self):
        # Test trianing a VAE for checking OOD
        o = self.get_test_ortho()
        c = self.get_test_coords()
        config = {'scale_data': True,
                  'learning'
                          # Whether to min max scale your data VAEs work best when data is pre-normalised & outliers removed for trainiing
                          'batch_norm': True,
                          'loss': {'loss_type': 'mse',  # mean squared error
                                   'distance_metric': 'mmd',  # Maximum mean discrepency (can use kl but it works worse)
                                   'mmd_weight': 0.001},
                          # Weight of mmd vs mse - basically having this > 1 will weigh making it normally distributed higher
                          # and making it < 1 will make reconstruction better.
                          'encoding': {'layers': [{'num_nodes': 2, 'activation_fn': 'selu'}]},  # First layer of encoding
                          'decoding': {'layers': [{'num_nodes': 2, 'activation_fn': 'selu'}]},  # Second layer of decoding
                          'latent': {'num_nodes': 1},
                          'optimiser': {'params': {'learning_rate': 0.001}, 'name': 'adam'}}  # Empty params means use default

        ood = OOD(o, c, config=config)
        mpp = 4
        dist = ood.train_ood(image=o, coords=c, bands=[o.get_band(b) for b in img_bands],
                             width_m=1000, height_m=1000)

        # Now let's also plot the reconstruction
        plt.hist(dist[:, 0])
        plt.show()
        vae = ood.vae
        encoding = vae.encode_new_data(ood.train_df[ood.training_cols].values, scale=True)
        plt.figure(figsize=(20, 2))
        n = 5
        for i in range(n):
            d = vae.decoder.predict(np.array([encoding[i]]))[0]
            ax = plt.subplot(1, n, i + 1)
            reshaped = d.reshape(ood.num_pix, ood.num_pix, 3) #(mpp*2)-1, (mpp*2)-1, 3)
            plt.imshow(reshaped)
        plt.show()

        encoding = ood.train_df[ood.training_cols].values # i.e. just do the data
        plt.figure(figsize=(20, 2))
        n = 5
        for i in range(n):
            d = vae.decoder.predict(np.array([encoding[i]]))[0]
            ax = plt.subplot(1, n, i + 1)
            reshaped = d.reshape((mpp*2)-1, (mpp*2)-1, 3)
            plt.imshow(reshaped)
        plt.show()

    def test_load_ood(self):
        # Load presaved model
        o = self.get_test_ortho()
        c = self.get_test_coords()
        ood = OOD(o, c)
        ood.load_saved_vae()
        print('loaded')

    def test_ood_predict(self):
        # Test trianing a VAE for checking OOD
        o = self.get_test_ortho()
        c = self.get_test_coords()
        ood = OOD(o, c)
        ood.load_saved_vae()
        print('loaded')
        # Now check prediction using waeldi ood
        drone_pine_coords = '../data/dryad_trees/location_files/ood.csv'
        oo_c = Coords(drone_pine_coords, x_col='Y', y_col='X', label_col='class',
                   id_col='id', sep=',', class1='OOD', class2='OOD')
        oo_c.transform_coords(tree_coords="EPSG:4326", image_coords="EPSG:32614", plot=True)
        # Build train df --> needs to be as above
        labels = [f'b{i}' for i in [1, 2, 3]]
        df, labels = ood.build_train_df(o, oo_c, bands=[o.get_band(b) for b in img_bands], max_pixel_padding=1,
                                        band_labels=labels)

        df['pred'], pred = ood.classify_ood(df[labels].values)
        print(df)
        plt.hist(pred[:, 0])
        plt.show()
        print(len(df[df['pred'] == True]))
        df, labels = ood.build_train_df(o, c, bands=[o.get_band(b) for b in img_bands],
                                        max_pixel_padding=1, band_labels=labels)
        df['pred'], pred = ood.classify_ood(df[labels].values)
        print(df)
        plt.hist(pred[:, 0])
        print(len(df[df['pred'] == True]))
        plt.show()
        labels = df['binary_label']
        correct = 0
        incorrect = 0
        for i, pred in enumerate(df['pred'].values):
            if pred and labels[i]:
                correct += 1
            elif not pred and labels[i] == 0:
                correct += 1
            else:
                incorrect += 1
        print(correct, incorrect, correct/(correct + incorrect))
        # Print the % OOD that were correct
        df.to_csv('ODD.csv')

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