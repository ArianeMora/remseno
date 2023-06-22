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

# Use Typer to autogenerate CLI for the required tools

"""
This contains the key functions to run modRunner.

# Ones that only require read files
1. Minimap2Runner, requires a (minimap2)
2. Eligos2Runner, requires a, 1 (eligos2)
3. DiffErrRunner, requires a, 1 (differr)

# Ones that require fast5 files
a. guppyRunner # mapping reads first step (guppy)
b. tombo # Shite, requires, a, 1 (tombo)
c. eventalign # requires a, 1 (eventalign)
d. nanocompore # requires a, 1, c (nanocompore)

"""
import sys
sys.path.append('/modrunnerLite')
from typing import Optional

import typer
from sciutil import SciUtil
import os

app = typer.Typer()
ERROR = 0
SUCCESS = 1


@app.command()
def run():
    print("done")


if __name__ == "__main__":
    app()
