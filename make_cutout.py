import numpy as np

#import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom

# Mostly shamelessly stolen from Sophie Reed (thanks)

def mad(l, med):
    return np.median(abs(l - med))


def make_cutout_lsst(coords, exp, coord_units="pixels", size=100, w_units="pixels", width=30.0):
    wcs = exp.getWcs()
    im = exp.maskedImage.image.array
    bbox = exp.getBBox()
    xoffset = bbox.getBeginX()
    yoffset = bbox.getBeginY()

    if len(coords) != 2:
        raise ValueError("len(coords) != 2")
    if coord_units == "radec":
        radec = afwGeom.SpherePoint(coords[0], coords[1], afwiGeom.degrees)
        pix = wcs.skyToPixel(coord)
    elif coord_units == "pixels":
        pix = coords
    else:
        raise ValueError("Unknown coordinate units " + coord_units)

    if w_units == "pixels":
        id1 = int(round(pix[0]))+size-xoffset
        id2 = int(round(pix[0]))-size-xoffset
        id3 = int(round(pix[1]))+size-yoffset
        id4 = int(round(pix[1]))-size-yoffset

        im_blank = np.zeros((size*2, size*2))
        ids = np.array([id1, id2, id3, id4])
        # print(ids)

    if w_units == "arcsecs":
        width = width/3600.0/2.0

        ra_w1 = ra - width/np.cos(np.deg2rad(dec))
        dec_w1 = dec - width
        ra_w2 = ra + width/np.cos(np.deg2rad(dec))
        dec_w2 = dec + width

        edge_coord1 = afwGeom.SpherePoint(ra_w1, dec_w1, afwGeom.degrees)
        edge_pix1 = wcs.skyToPixel(edge_coord1)
        x1 = int(edge_pix1[0] - xoffset)
        y1 = int(edge_pix1[1] - yoffset)
        edge_coord2 = afwGeom.SpherePoint(ra_w2, dec_w2, afwGeom.degrees)
        edge_pix2 = wcs.skyToPixel(edge_coord2)
        x2 = int(edge_pix2[0] - xoffset)
        y2 = int(edge_pix2[1] - yoffset)

        # print(edge_pix1[0]-xoffset, edge_pix2[0]-xoffset, edge_pix1[1]-yoffset, edge_pix2[1]-yoffset)

        if x1 > x2:
            x_width = x1 - x2
            id1 = x1
            id2 = x2
        else:
            x_width = x1 - x2
            id1 = x2
            id2 = x1
        if y1 > y2:
            y_width = y1 - y2
            id3 = y1
            id4 = y2
        else:
            y_width = y2 - y1
            id3 = y2
            id4 = y1
        if x_width > y_width:
            b_width = x_width
            id3 += (x_width-y_width)
        else:
            b_width = y_width
            id1 += (y_width-x_width)

        ids = np.array([id1, id2, id3, id4])
        # print(ids)

        im_blank = np.zeros((int(np.ceil(b_width)), int(np.ceil(b_width))))
    # print(im_blank.shape, id1-id2, id3-id4)


    if ids[3] < 0:
        c3 = -1*ids[3]
        ids[3] = 0
    else:
        c3 = 0
    if ids[2] > im.shape[0]:
        c2 = -1*(ids[2]-im.shape[0])
        ids[2] = im.shape[0]
    else:
        c2 = im_blank.shape[0]
    if ids[1] < 0:
        c1 = -1*ids[1]
        ids[1] = 0
    else:
        c1 = 0
    if ids[0] > im.shape[1]:
        c0 = -1*(ids[0]-im.shape[1])
        ids[0] = im.shape[1]
    else:
        c0 = im_blank.shape[0]

    im_cutout = im[ids[3]:ids[2], ids[1]:ids[0]]

    im_blank[c3:c2, c1:c0] = im_cutout

    med = np.median(im_cutout.flatten())
    im_mad = mad(im_cutout.flatten(), med)
    vmin = med - 2.0*1.4826*im_mad
    vmax = med + 5.0*1.4826*im_mad

    return (im_cutout, vmin, vmax, im_blank, ids, [c3, c2, c1, c0])


def MAD(l, med):
    import numpy as np
    return np.median(abs(l - med))


def cutout_scale(im, num_min=2.0, num_max=5.0):

    """
    Takes an image array and returns the vmin and vmax required to scale the image 
    between median + 5 * sigma MAD and median - 2 * sigma MAD
    """

    import numpy as np

    data = im.flatten()
    try:
        med = np.median(data[data != 0.0])
        sigma_MAD = 1.4826*MAD(data[data != 0.0], med)
    except IndexError:
        med = 0.0
        sigma_MAD = 0.0
    vmax = med + num_max * sigma_MAD
    vmin = med - num_min * sigma_MAD

    return vmin, vmax


def make_cutout(filename, RA, DEC, width, nhdu=0, w_units="arcsecs", verbose=False):

    """
    Makes cutouts from a file passed as either a filename or hdulist
    Returns a new hdulist with updated WCS and the cutout as the data
    The specified width should be in arcsecs or pixels.
    It is the full width.
    """

    from astropy import wcs
    # import wcsutil
    import astropy.io.fits as fits
    import astropy.io.fits.compression
    import numpy as np
    # import gc
    # import matplotlib.pyplot as plt
    import warnings
    from astropy.utils.exceptions import AstropyWarning
    import copy

    im_status = "good"

    if not verbose:
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        warnings.simplefilter('ignore', category=AstropyWarning)

    if isinstance(filename, str):
        with fits.open(filename) as h:
            hdr = h[nhdu].header
            data = h[nhdu].data
    else:
        try:
            hdr = filename[nhdu].header
            # It is this line data = filename[nhdu].data
            data = filename[nhdu].data
        except RuntimeError:
            print("File Corrupt")
            return None, "bad"
    try:
        test = hdr["NAXIS1"]
    except KeyError:
        return None, "bad"

    # Figure out how big the square would be at the centre
    test_pix = [[hdr["NAXIS1"]/2, hdr["NAXIS2"]/2]]

    try:
        from astropy import wcs
        w = wcs.WCS(hdr, naxis=2)

        if w_units == "arcsecs":
            width = width/3600.0/2.0
            w_coord = np.array([[RA - width/np.cos(np.deg2rad(DEC)), DEC - width], [RA + width/np.cos(np.deg2rad(DEC)), DEC + width]], np.float_)
            pix_coord = w.wcs_world2pix(w_coord, 1, ra_dec_order=True)
            coords = [int(pix_coord[0][0]), int(pix_coord[0][1]), int(pix_coord[1][0]), int(pix_coord[1][1])]
            test_coord = w.wcs_pix2world(test_pix, 1, ra_dec_order=True)
            test_edges = np.array([[test_coord[0][0], test_coord[0][1] - width],
                                  [test_coord[0][0], test_coord[0][1] + width]], np.float_)
            test_pix = w.wcs_world2pix(test_edges, 1, ra_dec_order=True)
            [[tx1, ty1], [tx2, ty2]] = test_pix

        elif w_units == "pixels":
            w_coord = np.array([[RA, DEC]], np.float_)
            pix_coord = w.wcs_world2pix(w_coord, 1)
            coords = [int(pix_coord[0][0]+width/2.0), int(pix_coord[0][1]-width/2.0), int(pix_coord[0][0]-width/2.0), int(pix_coord[0][1]+width/2.0)]
            # Create a background of zeros
            blank = np.zeros((width+1, width+1))
            [tx1, ty1, tx2, ty2] = coord

    except (UnboundLocalError, wcs.wcs.InconsistentAxisTypesError) as e:
        print("Axis type not supported")
        return None
    #    try:
    #        wcs = wcsutil.WCS(hdr)
    #    except KeyError:
    #        hdr = h[nhdu-1].header
    #        wcs = wcsutil.WCS(hdr)

    #    width = width/3600.0/2.0
    #    test_coord = wcs.image2sky(test_pix[0][0], test_pix[0][1])
    #    tx1, ty1 = wcs.sky2image(test_coord[0], test_coord[1] - width)
    #    tx2, ty2 = wcs.sky2image(test_coord[0], test_coord[1] + width)

    #    x1, y1 = wcs.sky2image(RA - width/np.cos(np.deg2rad(DEC)), DEC - width)
    #    x2, y2 = wcs.sky2image(RA + width/np.cos(np.deg2rad(DEC)), DEC + width)
    #    coords = [int(x1), int(y1), int(x2), int(y2)]

    if tx1 > tx2:
        x_width = tx1 - tx2
    else:
        x_width = tx1 - tx2
    if ty1 > ty2:
        y_width = ty1 - ty2
    else:
        y_width = ty2 - ty1
    if x_width > y_width:
        b_width = x_width
    else:
        b_width = y_width
    blank = np.zeros((int(np.ceil(b_width))+1, int(np.ceil(b_width))+1))

    # coords = [x1, y1, x2, y2]
    coords_clean = coords + []
    for (n, p) in enumerate(coords):
        if p < 0.0:
            coords_clean[n] = 0.0
            im_status = "bad"

        elif (p > hdr["NAXIS1"] and n % 2 == 0):
            coords_clean[n] = hdr["NAXIS1"]
            im_status = "bad"
        elif (p > hdr["NAXIS2"] and n % 2 != 0):
            coords_clean[n] = hdr["NAXIS2"]
            im_status = "bad"

    if len(data.shape) > 2:
        data = data[0, 0, :, :]
    if w_units == "pixels":
        im = data[coords_clean[1]:coords_clean[3], coords_clean[2]:coords_clean[0]]
    if w_units == "arcsecs":
        c1 = min([int(coords_clean[1]), int(coords_clean[3])])
        c2 = max([int(coords_clean[1]), int(coords_clean[3])])
        c3 = min([int(coords_clean[2]), int(coords_clean[0])])
        c4 = max([int(coords_clean[2]), int(coords_clean[0])])
        # im = data[int(coords_clean[1]):int(coords_clean[3]), int(coords_clean[2]):int(coords_clean[0])]
        im = data[c1:c2, c3:c4]

    if coords[0] > hdr["NAXIS1"]:
        by2 = hdr["NAXIS1"]-coords[2]
        im_status = "bad"
    else:
        by2 = im.shape[1]

    if coords[2] < 0:
        by1 = np.fabs(coords[2])
        im_status = "bad"
        by2 += by1
    else:
        by1 = 0

    if coords[3] > hdr["NAXIS2"]:
        bx2 = hdr["NAXIS2"]-coords[1]
        im_status = "bad"
    else:
        bx2 = im.shape[0]

    if coords[1] < 0:
        bx1 = np.fabs(coords[1])
        im_status = "bad"
        bx2 += bx1
    else:
        bx1 = 0

    if (coords_clean[1] == coords_clean[3]) or (coords_clean[0] == coords_clean[2]):
        im = blank
        im_status = "bad"

    else:
        try:
            blank[int(bx1):int(bx2), int(by1):int(by2)] = im
        except ValueError:
            # blank = np.zeros((max(im.shape), max(im.shape)))
            blank = np.zeros((max([bx2, by2]), max([bx2, by2])))
            blank[int(bx1):int(bx2), int(by1):int(by2)] = im
        im = blank

    if im_status == "good" and w_units == "arcsecs":
        im = data[c1:c2, c3:c4]

    phdu = fits.PrimaryHDU()
    h2 = copy.deepcopy(hdr)
    h2["CRPIX1"] = hdr["CRPIX1"]-coords[2]
    h2["CRPIX2"] = hdr["CRPIX2"]-coords[1]
    imhdu = fits.ImageHDU(header=h2, data=im)
    hdulist = fits.HDUList([phdu, imhdu])

    return hdulist, im_status


def cutout_HST(RA, DEC, sigma=0, width=30.0, type="png", return_data=False):

    """
    Makes cutouts from the HST image of the COSMOS field
    Width is the full width and needs to be given in arcsecs
    Can return a fits file or a png
    Can also return just the array used to make the image and the scaling parameters
    sigma applies a gaussian filter
    """

    import astropy.io.fits as fits
    from astropy import wcs
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    datas = []
    with open("/scratch/sr525/hstCosmosImages/tiles/corners.txt", "r") as f:
        for line in f:
            l = line.split(",")
            ra_min = float(l[0])
            ra_max = float(l[2])
            dec_min = float(l[1])
            dec_max = float(l[3])
            filename = "/scratch/sr525/hstCosmosImages/" + l[4][38:]
            if RA < ra_min and RA > ra_max and DEC > dec_min and DEC < dec_max:
                print(filename)
                with fits.open(filename[:-1]) as h:
                    hdulist, im_status = make_cutout(h, RA, DEC, width, nhdu=0)
                    im = hdulist[1].data
                    vmin, vmax = cutout_scale(im)
                    if sigma > 0:
                        im = gaussian_filter(im, sigma)
                    if not return_data:
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.axes.get_xaxis().set_visible(False)
                        ax.axes.get_yaxis().set_visible(False)
                        ax.imshow(im, vmax=vmax, vmin=vmin)
                        fig = plt.gcf()
                        fig.set_size_inches(2.5, 2.5)

                    if return_data:
                        datas.append((hdulist, vmin, vmax))
                        return datas
                    else:
                        return fig
