'Calculates indices from tiff file.'

from remseno import *


def get_sr(image, nir_band: int, red_band: int):
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
    sr = (nir - red_band)

    return sr


######################################################################################

"""
    Calculates NDVI

    :param nir_band: Near Infrared Band
    :param red_band: Red band
    :return: ndvi
"""


def get_ndvi(image, nir_band: int, red_band: int):
    # Get the bands from the image

    nir = image.read(nir_band)
    red_band = image.read(red_band)
    ndvi = (nir - red_band) / (nir + red_band)

    return ndvi


######################################################################################
"""
    Calculates Red edge reflectance parameter

    :param nir_band: Near Infrared Band
    :param red_band: Red band
    :param r_edge: red edge (705)
    :return: redge
"""


def get_redge(image, nir_band: int, r_edge: int, green: int):
    # Get the bands from the image

    nir = image.read(nir_band)
    r_edge = image.read(r_edge)
    green = image.read(green)
    redge = nir / green * r_edge

    return redge


######################################################################################
"""
    Calculates Red edge reflectance parameter 2

    :param nir_band: Near Infrared Band
    :param red_band: Red band
    :param r_edge: red edge (705)
    :return: redge
"""


def get_redge2(image, red_band: int, r_edge: int, green: int):
    # Get the bands from the image

    red_band = image.read(red_band)
    r_edge = image.read(r_edge)
    green = image.read(green)
    redge2 = red_band / green * r_edge

    return redge2


######################################################################################

"""
    Calculates simple ratio Red edge reflectance parameter 3

    :param red_band: Red band
    :param r_edge: red edge (705)
    :return: siredge
"""


def get_siredge(image, red_band: int, r_edge: int):
    # Get the bands from the image

    red_band = image.read(red_band)
    r_edge = image.read(r_edge)
    siredge = red_band / r_edge

    return siredge


######################################################################################

"""
    Calculates Normalized Greenness (Norm G)

    :param red_band: Red band
    :param green_band: Green band
    :param blue_band: Blue band
    :return: normg
"""


######################################################################################

def get_normg(image, red_band: int, blue_band: int, green_band):
    # Get the bands from the image

    red_band = image.read(red_band)
    green_band_edge = image.read(green_band)
    blue_band_edge = image.read(blue_band)
    normg = green_band / (green_band + blue_band + red_band)

    return normg


######################################################################################
"""
    Calculates MCARI 

    :param red_band: Red band
    :param green_band: Green band
    :param blue_band: Blue band
    :return: normg
"""


def get_mcari(image, red_band: int, blue_band: int, green_band):
    # Get the bands from the image

    red_band = image.read(red_band)
    green_band_edge = image.read(green_band)
    blue_band_edge = image.read(blue_band)
    normg = green_band / (green_band + blue_band + red_band)

    return normg

######################################################################################
