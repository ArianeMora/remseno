'Calculates indices from tiff file.'
import math

from tensorflow_probability.substrates import numpy

'The indices are modified from ' \
'Wang, Z., Wang, T., Darvishzadeh, R., Skidmore, A. K., Jones, S., Suarez, L., ... & Hearne, J. (2016). ' \
'Vegetation indices for mapping canopy foliar nitrogen in a mixed temperate forest. ' \
'Remote sensing, 8(6), 491.'
'From Table 1. The indices are developed with the available indices from PlanetScope.' \
'For the bands that were not available, the indices were not calculated, for the other, the closest band has been considered.'

from remseno import *
import numpy as np


def get_all_planetscope(img):
    nitian = get_nitian(image=img, r_edge=7, blue_band=2)
    ndvi = get_ndvi(image=img, red_band=6, nir_band=8)
    sr = get_sr(image=img, red_band=6, nir_band=8)
    tvi = get_tvi(image=img, red_band=6, rededge_band=7, green_band=4)
    #rdvi = get_rdvi(image=img, red_band=6, nir_band=8)
    gi = get_gi(image=img, red_band=6, green_band=4)
    gndvi = get_gndvi(image=img, green_band=4, nir_band=8)
    pri = get_pri(image=img, green_band=4, greeni_band=3)
    osavi = get_osavi(image=img, red_band=6, nir_band=8)
    tcari = get_tcari(image=img, rededge_band=7, greeni_band=3, red_band=6)
    redge = get_redge(image=img, nir_band=8, green=4, r_edge=7)
    redge2 = get_redge2(image=img, red_band=6, green=4, r_edge=7)
    siredge = get_siredge(image=img, red_band=6, r_edge=7)
    normg = get_normg(image=img, red_band=6, green_band=4, blue_band=2)
    schl = get_schl(image=img, red_band=6, rededge_band=7, nir_band=8)
    schlcar = get_schlcar(image=img, red_band=6, greeni_band=3)
    return nitian, ndvi, sr, tvi, gi, gndvi, pri, osavi, tcari, redge, redge2, siredge, normg, schl, schlcar

"""
    Calculates NI_Tian - Nitrogen index

    Ref: Tian, Y.C.; Yao, X.; Yang, J.; Cao, W.X.; Hannaway, D.B.; Zhu, Y. 
    Assessing newly developed and published vegetation indices for estimating 
    rice leaf nitrogen concentration with ground- and space-based hyperspectral 
    reflectance. Field Crops Res. 2011, 120, 299–310.

    (The original formula uses two red edge bands. In PSSD we have only one,
    thus one has been used.) 

    :param red_edge: Red edge band
    :param blue_band: Blue band
    :return: nitian
"""


def get_nitian(image, r_edge: int, blue_band: int):
    # Get the bands from the image

    r_edge = image.read(r_edge)
    blue_band = image.read(blue_band)
    nitian = r_edge / (r_edge + blue_band)

    return nitian


######################################################################################

"""
    Calculates NDVI

    Tucker, C.J. Red and photographic infrared linear combinations 
    for monitoring vegetation. 
    Remote Sens. Environ. 1979, 8, 127–150.

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
    sr = (nir / red_band)

    return sr


######################################################################################


def get_tvi(image, rededge_band: int, red_band: int, green_band: int):
    """
    Gets the TVI (triangular vegetation index) 0.5(120(red_edge-green)-200(green+red))
    Structural index
    Broge, N. H., & Leblanc, E. (2001).
    Comparing prediction power and stability of broadband and
    hyperspectral vegetation indices for estimation of green leaf area index and
    canopy chlorophyll density.
    Remote sensing of environment, 76(2), 156-172.

    :param rededge_band: Red Edge Band
    :param red_band: Red band
    :param green_band: Green band
    :return: tvi
    """
    # Get the bands from the image

    rededge_band = image.read(rededge_band)
    red_band = image.read(red_band)
    green_band = image.read(green_band)
    tvi = (0.5 * (120 * (rededge_band - green_band)) - (200 * (red_band + green_band)))

    return tvi


######################################################################################

def get_rdvi(image, nir_band: int, red_band: int):
    """
    Gets the RDVI (renormalized difference vegetation index) (nir-red) / (sqrt (nir+red))
    Structural index

    Roujean, J.L.; Breon, F.M.
    Estimating PAR absorbed by vegetation from bidirectional reflectance measurements.
    Remote Sens. Environ. 1995, 51, 375–384.

    :param nir_band: NIR Band
    :param red_band: Red band
    :return: rdvi
    """
    # Get the bands from the image

    red_band = image.read(red_band)
    nir_band = image.read(nir_band)
    rdvi = (nir_band - red_band) / math.sqrt(nir_band + red_band)

    return rdvi


######################################################################################

def get_gi(image, green_band: int, red_band: int):
    """
    Gets the Green Index witu Simple ratio values for this: green/red
    Canopy Chlorophyll Index
    Smith, R.; Adams, J.; Stephens, D.; Hick, P. Forecasting wheat yield
    in a Mediterranean-type environment from the NOAA satellite.
    Aust. J. Agric. Res. 1995,

    :param green_band: green Band
    :param red_band: Red band
    :return: gi
    """
    # Get the bands from the image

    green_band = image.read(green_band)
    red_band = image.read(red_band)
    gi = (green_band - red_band)

    return gi


######################################################################################

"""
    Calculates GNDVI
    Canopy Chlorophyll Index
    GREEN NDVI
    Gitelson, A.A.; Kaufman, Y.J.; Merzlyak, M.N. 
    Use of a green channel in remote sensing of global vegetation from EOS-MODIS. 
    Remote Sens. Environ. 1996, 58, 289–298. 


    :param nir_band: Near Infrared Band
    :param green_band: Green band
    :return: gndvi
"""


def get_gndvi(image, nir_band: int, green_band: int):
    # Get the bands from the image

    nir = image.read(nir_band)
    green_band = image.read(green_band)
    gndvi = (nir - green_band) / (nir + green_band)

    return gndvi


######################################################################################

"""
    Calculates PRI
    photochemical reflectance index

    Gamon, J. A., Penuelas, J., & Field, C. B. (1992). 
    A narrow-waveband spectral index that tracks diurnal changes in photosynthetic efficiency. 
    Remote Sensing of environment, 41(1), 35-44.

    D'Odorico, P., Schönbeck, L., Vitali, V., Meusburger, K., Schaub, M., Ginzler, C., ... & Ensminger, I. (2021). 
    Drone‐based physiological index reveals long‐term acclimation and drought stress responses in trees. 
    Plant, Cell & Environment, 44(11), 3552-3570.


    :param greeni_band: Green I band
    :param green_band: Green band
    :return: pri
"""


def get_pri(image, greeni_band: int, green_band: int):
    # Get the bands from the image

    greeni_band = image.read(greeni_band)
    green_band = image.read(green_band)
    pri = (greeni_band - green_band) / (greeni_band + green_band)

    return pri


######################################################################################

"""
    Calculates OSAVI
    Optimization of soil-adjusted vegetation indices

    Canopy Chlorophyll Index

    Rondeaux, G.; Steven, M.; Baret, F. 
    Optimization of soil-adjusted vegetation indices. 
    Remote Sens. Environ. 1996, 55, 95–107.


    :param nir_band: NIR band
    :param red_band: Red band
    :return: osavi
"""


def get_osavi(image, nir_band: int, red_band: int):
    # Get the bands from the image

    nir_band = image.read(nir_band)
    red_band = image.read(red_band)
    osavi = (nir_band - red_band) / (nir_band + red_band + 0.16)

    return osavi


######################################################################################
"""
    Calculates TCARI
    Transformed Chlorophyll Absorption in ReflectanceIndex

    Canopy Chlorophyll Index

    Haboudane, D., Miller, J. R., Tremblay, N., Zarco-Tejada, P. J., & Dextraze, L. (2002). 
    Integrated narrow-band vegetation indices for prediction of crop chlorophyll content for 
    application to precision agriculture. 
    Remote sensing of environment, 81(2-3), 416-426.


    :param rededge_band: rededge band
    :param red_band: Red band
    :param greeni_band: Green I band
    :return: tcari
"""


def get_tcari(image, rededge_band: int, red_band: int, greeni_band: int):
    # Get the bands from the image

    rededge_band = image.read(rededge_band)
    red_band = image.read(red_band)
    greeni_band = image.read(greeni_band)
    tcari = 3 * (((rededge_band - red_band) - 0.2 * (rededge_band - greeni_band)) * (rededge_band - red_band))

    return tcari


######################################################################################

"""
    Calculates Red edge reflectance parameter

    Datt, B. (1998). Remote sensing of chlorophyll a, chlorophyll b, 
    chlorophyll a+ b, and total carotenoid content in eucalyptus leaves. 
    Remote Sensing of Environment, 66(2), 111-121.

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

    Datt, B. (1998). Remote sensing of chlorophyll a, chlorophyll b, 
    chlorophyll a+ b, and total carotenoid content in eucalyptus leaves. 
    Remote Sensing of Environment, 66(2), 111-121.

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


def get_normg(image, red_band: int, blue_band: int, green_band):
    # Get the bands from the image

    red_band = image.read(red_band)
    green_band = image.read(green_band)
    blue_band = image.read(blue_band)
    normg = green_band / (green_band + blue_band + red_band)

    return normg


######################################################################################


def get_schl(image, nir_band: int, red_band: int, rededge_band: int):
    """
    Gets the Leaf Chlorophyll Index: (nir - rededge) / (nir + red)
    Leaf Chlorophyll Index

    le Maire, G., François, C., Dufrˆene, E., 2004.
    Towards universal broad leaf chlorophyll
    indices using PROSPECT simulated database and hyperspectral
    reflectance measurements. Remote Sens. Environ

    :param rededge_band: red egde Band
    :param red_band: Red band
    :param nir_band: nir band
    :return: schl
    """
    # Get the bands from the image

    nir_band = image.read(nir_band)
    rededge_band = image.read(rededge_band)
    red_band = image.read(red_band)
    schl = (nir_band - rededge_band) / (nir_band + red_band)

    return schl


######################################################################################

"""
    Calculates schlcar

    Chlorophyll Carotenoids ratio (sChlCar) 

    Gamon, J.A., Huemmrich, K.F., Wong, C.Y.S., Ensminger, I., Garrity, S., Hollinger, D.Y., Pe˜nuelas, J., 2016. 
    A remotely sensed pigment index reveals photosynthetic phenology in evergreen conifers. 
    Proc. Natl. Acad. Sci. 113 (46), 13087. 

    :param greeni_band: Green I Band
    :param red_band: Red band
    :return: schlcar
"""


def get_schlcar(image, greeni_band: int, red_band: int):
    # Get the bands from the image

    greeni_band = image.read(greeni_band)
    red_band = image.read(red_band)
    schlcar = (greeni_band - red_band) / (greeni_band + red_band)

    return schlcar


######################################################################################

'Opens the imagery, loads the bands and plosts the results'
