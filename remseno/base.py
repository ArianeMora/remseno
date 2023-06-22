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
from datetime import datetime
import os
import subprocess
import time
from sciutil import SciUtil
# Imports and things needed across everything
import os
from sciutil import *
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from tqdm import tqdm

import pandas as pd
from pyproj import Transformer


class Remsenso:

    def __init__(self, verbose=True, debug=False, processes=1, logging_file=None, name=""):
        self.u = SciUtil()
        self._debug = debug
        self._processes = processes
        self._verbose = verbose
        self._logging_file = logging_file
        self.name = name # This is just used for logging
        self.ortho = None
        self.coord = None

    def run_cmd(self, command):
        # Check if there is a > or | since we'll change it to a string
        # possibly hacky.
        if isinstance(command, list):
            if '>' in command or '|' in command:
                command = ' '.join(command)
        if self._debug:
            self.printv(command)
        else:
            # Keep track of the time and resource usage
            if self._logging_file:
                start_time = time.time()
                run_log = self._logging_file.replace('.txt', "_" + self.name + '_' + datetime.now().strftime(
                    "%Y%b%d%H%M%S%f") + '.txt')
                if isinstance(command, str):
                    os.system(f'/usr/bin/time -v {command} 2> {run_log}')
                else:
                    os.system(f'/usr/bin/time -v {" ".join(command)} 2> {run_log}')
                with open(self._logging_file, 'a+') as f:
                    if isinstance(command, str):
                        f.write(f'{self.name}\t{(time.time() - start_time)}\t{run_log}\t{command}\n')
                    else:
                        f.write(f'{self.name}\t{(time.time() - start_time)}\t{run_log}\t{" ".join(command)}\n')

            else:
                if isinstance(command, str):
                    os.system(command)
                elif isinstance(command, list):
                    subprocess.run(command)
                else:
                    print("MUST BE LIST OR STRING")

    def printv(self, *args):
        if self._verbose:
            print(*args)
        return