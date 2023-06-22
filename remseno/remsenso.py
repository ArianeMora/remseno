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

"""
Class that combines remote sensing data and builds a classifier on it
"""

from ortho import Ortho
from coord import Coords


class Remsenso:

    def __init__(self):
        print('init')
        self.orthos = []
        self.coords = None

    def load_orthos(self, ortho_list: list):
        """
        Load a list of orthomosaics.

        :param ortho_list:
        :return:
        """
        for ortho_path in ortho_list:
            ortho = Ortho()
            self.orthos.append(ortho.load_ortho(ortho_path))

    def view_ortho(self, ortho_id=None, band=1):
        """
        View either all orthds
        :param ortho_id: the ID of the ortho (or all if left as None)
        :param band: the band to show
        :return:
        """
        if ortho_id:
            self.orthos[ortho_id].plot(band)
        else:
            for o in self.orthos:
                o.plot(band)

