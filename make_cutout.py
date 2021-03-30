# This file is part of modelling_research.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import astropy.io.fits as fits
from astropy.wcs import WCS
import glob
from lsst.geom import degrees, Point2D
import multiprofit.objects as mpfObj
import numpy as np
import os


def find_boxes_overlapping(left, right, bottom, top, corners_boxes, first=True):
    """Find the set of overlapping boxes given a single box and a list of other boxes to compare.

    Parameters
    ----------
    left, right, bottom, top : `float`
        Coordinates of the edges of the box to test for overlap.
    corners_boxes : iterable of array-like
        An iterable of box edge/corner coordinates as above (left, right, bottom, top).
    first : `bool`
        Whether to return the first match only.

    Returns
    -------
    idx : `int` or set [`int`]
        The index of the first matching box if first is True, or the set of all matching box indices if not.
    """
    found = set()
    for idx, (box_left, box_right, box_bottom, box_top) in enumerate(corners_boxes):
        if not (right < box_left or left > box_right or top < box_bottom or bottom > box_top):
            if first:
                return idx
            found.add(idx)
    return None if first else found


def get_corners_src(src, wcs, field_centroid=None):
    """Get the sky coordinates of corners of a source's bounding box in degrees.

    Parameters
    ----------
    src : `lsst.afw.table.BaseRecord`
        A source to extract a cutout for.
    wcs : `lsst.afw.geom.skyWcs`
        The WCS for the source record.
    field_centroid : `str`
        The centroid field to use; default 'slot_Centroid'.

    Returns
    -------
    corners : `list`
        A list of ra, dec coordinates for each corner, in degrees.

    """
    if field_centroid is None:
        field_centroid = 'slot_Centroid'
    pixel_src = Point2D([src[f'{field_centroid}_{ax}'] for ax in ['x', 'y']])
    cens = wcs.pixelToSky(pixel_src).getPosition(degrees)
    bbox = src.getFootprint().getBBox()
    corners = wcs.pixelToSky([Point2D(x) for x in (bbox.getMin(), bbox.getMax())])
    corners = [x.getPosition(degrees) for x in corners]
    return corners, cens


def get_corners_exposure(exposure):
    """Get the corners of an exposure in degrees.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        An exposure with WCS.

    Returns
    -------
    ra_corner : `list` [`float`]
        Right ascension of each corner in degrees.
    dec_corner : `list` [`float`]
        Declination of each corner in degrees.
    """
    wcs_exp = exposure.getWcs()
    ra_corner = []
    dec_corner = []
    for corner in exposure.getBBox().getCorners():
        ra, dec = wcs_exp.pixelToSky(Point2D(corner))
        ra_corner.append(ra.asDegrees())
        dec_corner.append(dec.asDegrees())
    return ra_corner, dec_corner


def get_exposure_cutout_HST(corners, cens, exposures_hst, get_inv_var=False, get_psf=False):
    """Get a cutout of the COSMOS HST data covering a given region, if any.

    Parameters
    ----------
    corners : `iterable`
        x, y coordinates of points (corners) defining cutout region.
    cens : `iterable`
        ra, dec sky coordinates to transform to pixel coordinates.
    exposures_hst : `list` [`multiprofit.objects.Exposure`]
        A list of HST exposures, as returned by `get_exposures_HST_COSMOS`.
    get_inv_var : `bool`
        Whether to retrieve the inverse variance map.
    get_psf : `bool`
        Whether to retrieve the PSF image.

    Returns
    -------
    exposure_cutout : `multiprofit.objects.Exposure`
        A MultiProFit exposure object of the cutout region, including image and inverse variance.
    cen_hst : `list` [`float`]
         x, y pixel coordinates of `cens`.
    psf : `numpy.ndarray` or None
        The PSF image if get_psf else None.
    """
    x = None
    for exposure in exposures_hst:
        wcs_hst = exposure.meta['wcs']
        array_shape = wcs_hst.array_shape
        x, y = [], []
        for corner in corners:
            pixel_hst = wcs_hst.world_to_array_index_values([corner])[0]
            # TODO: Verify that axes are in the correct order
            if 0 < pixel_hst[0] < array_shape[0] and 0 < pixel_hst[1] < array_shape[1]:
                x.append(pixel_hst[0])
                y.append(pixel_hst[1])
            else:
                x = None
                break
        if x is not None:
            x = (np.min(x), np.max(x))
            y = (np.min(y), np.max(y))
            break
    if x is None:
        raise RuntimeError(f"Couldn't get COSMOS HST F814W cutout")
    images = [exposure.image]
    if get_inv_var:
        images.append(exposure.get_var_inverse())
    cutouts = [np.float64(img[y[0]:y[1], x[0]:x[1]]) for img in images]
    cutout_hst = cutouts[0]
    cen_hst = wcs_hst.world_to_array_index_values([cens])[0] - np.array((x[0], y[0]))
    if get_inv_var:
        cutout_hst_weight = cutouts[1]
        bg_hst = cutout_hst < 0
        var_hst = np.mean(cutout_hst[bg_hst]**2)
        error_inverse = cutout_hst_weight / (var_hst * np.median(cutout_hst_weight[bg_hst]))
    if get_psf:
        cat_rg_within = exposure.meta['cosmos_cat_rg_within']
        closest = cat_rg_within[
            np.argmin((cat_rg_within['RA']-cens[0])**2 + (cat_rg_within['DEC']-cens[1])**2)]
        files_psf = exposure.meta['cosmos_cat_rg_psf_files']
        psf = np.float64(files_psf[closest['PSF_FILENAME']][closest['PSF_HDU']].data)
    else:
        psf = None
    exposure_cutout = mpfObj.Exposure(
        band=exposure.band, image=cutout_hst, error_inverse=error_inverse if get_inv_var else None,
        is_error_sigma=False)
    exposure_cutout.meta['wcs'] = wcs_hst
    return exposure_cutout, cen_hst, psf


def get_exposures_HST_COSMOS(ra_corner, dec_corner, tiles_cosmos, path_cosmos_galsim=None):
    """Get the COSMOS HST-F814W data overlapping the sky coordinates, if any.

    Parameters
    ----------
    ra_corner, dec_corner : array-like
        Right ascension/declination arrays, the extrema of which will determine the cutout coordinates.
    tiles_cosmos: `list`
        A list of COSMOS tiles as returned by `get_tiles_HST_COSMOS`.
    path_cosmos_galsim: `str`
        A path to the COSMOS GalSim catalog; see `fit` for details.

    Returns
    -------
    exposures : `list` [`multiprofit.objects.Exposure`]
        MultiProFit exposures with the image data, inverse variance and WCS of overlapping data.
    """
    tiles_hst = get_tiles_overlapping_HST(
        [np.min(ra_corner), np.max(ra_corner)], [np.min(dec_corner), np.max(dec_corner)], tiles_cosmos,
        first=False)
    exposures = []
    has_cosmos_galsim = path_cosmos_galsim is not None
    if has_cosmos_galsim:
        cat_rg = fits.open(os.path.join(path_cosmos_galsim, "real_galaxy_catalog_25.2.fits"))[1].data
        ra, dec = cat_rg['RA'], cat_rg['DEC']
        cosmos_cat_rg_psf_files = {}
        cosmos_cat_rg_psf_filenames = set()
    for image, error_inverse, corners in tiles_hst:
        exposure = mpfObj.Exposure(
            image=image[0].data, error_inverse=error_inverse[0].data, is_error_sigma=False, psf=None,
            band='F814W')
        if has_cosmos_galsim:
            within = (ra > corners[0]) & (ra < corners[1]) & (dec > corners[2]) & (dec < corners[3])
            cat_rg_within = cat_rg[within]
            cosmos_cat_rg_psf_filenames.update(cat_rg_within['PSF_FILENAME'])
            exposure.meta['cosmos_cat_rg_within'] = cat_rg_within
            exposure.meta['cosmos_cat_rg_psf_files'] = cosmos_cat_rg_psf_files
        exposure.meta['wcs'] = WCS(image[0])
        exposures.append(exposure)
    if has_cosmos_galsim:
        for filename in cosmos_cat_rg_psf_filenames:
            cosmos_cat_rg_psf_files[filename] = fits.open(os.path.join(path_cosmos_galsim, filename))
    return exposures


def get_tiles_overlapping_HST(ra, dec, tiles, width=None, height=None, first=False):
    """Get the tiles containing a skybox.

    Parameters
    ----------
    ra, dec : `float` or array-like of `float`
        Coordinates in degrees of a point or edges of a box to obtain a cutout for.
    tiles : `list`
        List of HST tiles as returned by `get_tiles_HST_COSMOS`.
    width, height : `float`
        The full width and height of the cutout in arcseconds. Default 30" if neither specified, else equal if
        only one specified.
    first : `bool`
        Whether to return the only the first cutout or all, if multiple are found.

    Returns
    -------
    rval : `list`
        A (shorter) list of the tiles overlapping

    """
    if width is None and height is None:
        width, height = 1/120., 1/120.
    elif width is None:
        width = height
    elif height is None:
        height = width

    is_scalar = np.isscalar(ra), np.isscalar(dec)
    if is_scalar[0] != is_scalar[1]:
        raise RuntimeError(f'Inconsistent inputs: is_scalar(RA, DEC) = ({is_scalar[0]}, {is_scalar[1]})')
    if is_scalar[0]:
        half = width/2
        ra_box = ra - half, ra + half
        half = height/2
        dec_box = dec - half, dec + half
    else:
        ra_box, dec_box = ra, dec
    boxes = find_boxes_overlapping(ra_box[0], ra_box[1], dec_box[0], dec_box[1], [x[2] for x in tiles],
                                   first=first)
    if first:
        return tiles[boxes]
    return [tiles[box] for box in boxes]


def get_tiles_HST_COSMOS(path=None):
    """Get fits objects and skyboxes for all of the HST COSMOS tiles.

    Parameters
    ----------
    path : `str`
        A path to a directory containing the COSMOS FITS images.

    Returns
    -------
    files : `list`
        A list of tuples containing:
            image : `astropy.io.fits.HDUList`
                The science image.
            weights : `astropy.io.fits.HDUList` or None
                The inverse variance map if found else None.
            corners : `tuple` [`float`]
                ra_min, ra_max, dec_min, dec_max of the tile in degrees.

    Notes
    -----
    The COSMOS data can be downloaded from https://irsa.ipac.caltech.edu/data/COSMOS/images/.
    You will likely want to use acs_mosaic_2.0 because these are aligned N-S W-E; but if you care to deal
    with the ~11 degree rotation of the fields yourself, the unrotated versions are in acs_2.0.
    """

    if path is None:
        path = os.path.join(os.path.sep, 'project', 'sr525', 'hstCosmosImages', 'tiles')
    filenames = glob.glob(os.path.join(path, 'acs_I_030mas_*_sci.fits'))
    tiles = []
    for filename in filenames:
        image = fits.open(filename)
        weights = f'{filename[:-8]}wht.fits'
        wcs = WCS(image[0])
        corners = wcs.calc_footprint()
        corners = (np.min(corners[:, 0]), np.max(corners[:, 0]), np.min(corners[:, 1]), np.max(corners[:, 1]))
        tiles.append((image, fits.open(weights) if os.path.exists(weights) else None, corners))
    return tiles
