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
from tqdm import tqdm
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

""" Generates different spectral indices from any .tiff file.
Be aware of the band order."""

from remseno.base import Remsenso


class Index(Remsenso):

    def __init__(self):
        super().__init__()
        self.ortho = None

    def get_bands(self):
        """
        Get all the bands in an image
        :return:
        """
        return self.ortho.count

    def get_sr(self, image, nir_band: int, red_band: int):
        """
        Gets the Simple ratio values for this: Nir/red
        The green foliage status of a canopy
        Rouse, J. W., Haas, R. H.,Schell, J. A.and Deering, D. W.(1973).Monitoring
        vegetation systems in the Great Plains with ERTS. Third ERTS Symposium, NASA SP-351 I, pp. 309–317.

        :param nir_band: Near Infrared Band
        :param red_band: Red band
        :return: sr
        """
        # Get the bands from the image
        nir = image.read(nir_band)
        red_band = image.read(red_band)
        sr = (nir-red_band)/(nir+red_band)
        return sr

