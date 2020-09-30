#!/usr/bin/env python
# coding: utf-8

# # Benchmarking convolution
# 
# This notebook benchmarks several packages' implementations of direct and Fourier transform-based convolution using galaxy- and PSF-like images. It also compares against evaluation of a Gaussian mixture approximation with analytic convolution using MultiProFit.

# In[1]:


import astropy.convolution as apconv
import galsim as gs
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
import timeit
from timeit import default_timer as timer


# In[2]:


try:
    import cv2
    cv2_all = True
except Exception as error:
    print(f'Failed to import cv2: {error}')
    cv2_all = False
try:
    import multiprofit as mpf
    import multiprofit.fitutils as mpfFit
    import multiprofit.objects as mpfObj
    mpf_draw = True
except Exception as error:
    print(f'Failed to import multiprofit: {error}')
    mpf_draw = False
try:
    import pyprofit as profit
    profit_all = True
except Exception as error:
    print(f'Failed to import pyprofit: {error}')
    profit_all = False
try:
    import pyfftw
    pyfftw_all = True
except Exception as error:
    print(f'Failed to import pyfftw: {error}')
    pyfftw_all = False
try:
    from scarlet.observation import convolve
    from scarlet.interpolation import get_filter_coords, get_filter_bounds
    scarlet_direct = True
except Exception as error:
    scarlet_direct = False
    print(f'Failed to import scarlet direct: {error}')
try:
    from scarlet import fft
    scarlet_fft = True
except Exception as error:
    print(f'Failed to import scarlet fft: {error}')
    scarlet_fft = False


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
#sns.set_style('dark')
mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['image.origin'] = 'lower'
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.facecolor'] = 'w'


# In[4]:


def imshow(imgs, show=True, title=None):
    fig, axes = plt.subplots(1, len(imgs))
    for axis, img in zip(axes, imgs):
        axis.imshow(img)
    if title:
        fig.suptitle(title, y=0.7)
    if show:
        plt.show(block=False)
    return fig, axes


# Astropy convolve with minor efficiency boosts
# 
# - Add option to return the kernel fft for re-use
# - Slightly optimize the 'fill' nan_treatment

# In[5]:


from astropy import units as u

def _copy_input_if_needed(input, dtype=float, order='C', nan_treatment=None,
                          mask=None, fill_value=None):
    # strip quantity attributes
    if hasattr(input, 'unit'):
        input = input.value
    output = input
    # Copy input
    try:
        # Anything that's masked must be turned into NaNs for the interpolation.
        # This requires copying. A copy is also needed for nan_treatment == 'fill'
        # A copy prevents possible function side-effects of the input array.
        if nan_treatment == 'fill' or np.ma.is_masked(input) or mask is not None:
            if np.ma.is_masked(input):
                # ``np.ma.maskedarray.filled()`` returns a copy, however there
                # is no way to specify the return type or order etc. In addition
                # ``np.nan`` is a ``float`` and there is no conversion to an
                # ``int`` type. Therefore, a pre-fill copy is needed for non
                # ``float`` masked arrays. ``subok=True`` is needed to retain
                # ``np.ma.maskedarray.filled()``. ``copy=False`` allows the fill
                # to act as the copy if type and order are already correct.
                output = np.array(input, dtype=dtype, copy=False, order=order, subok=True)
                output = output.filled(fill_value)
            else:
                # Since we're making a copy, we might as well use `subok=False` to save,
                # what is probably, a negligible amount of memory.
                output = np.array(input, dtype=dtype, copy=True, order=order, subok=False)

            if mask is not None:
                # mask != 0 yields a bool mask for all ints/floats/bool
                output[mask != 0] = fill_value
        else:
            # The call below is synonymous with np.asanyarray(array, ftype=float, order='C')
            # The advantage of `subok=True` is that it won't copy when array is an ndarray subclass. If it
            # is and `subok=False` (default), then it will copy even if `copy=False`. This uses less memory
            # when ndarray subclasses are passed in.
            output = np.array(input, dtype=dtype, copy=False, order=order, subok=True)
    except (TypeError, ValueError) as e:
        raise TypeError('input should be a Numpy array or something '
                        'convertible into a float array', e)
    return output


def convolve_fft(array, kernel, boundary='fill', fill_value=0.,
                 nan_treatment='interpolate', normalize_kernel=True,
                 normalization_zero_tol=1e-8,
                 preserve_nan=False, mask=None, crop=True, return_fft=False,
                 return_kernfft=False, kernfft=None,
                 fft_pad=None, psf_pad=None, min_wt=0.0, allow_huge=False,
                 fftn=np.fft.fftn, ifftn=np.fft.ifftn,
                 complex_dtype=complex):
    # Check kernel is kernel instance
    if nan_treatment not in ('interpolate', 'fill'):
        raise ValueError("nan_treatment must be one of 'interpolate','fill'")

    # Convert array dtype to complex
    # and ensure that list inputs become arrays
    array = _copy_input_if_needed(array, dtype=complex, order='C',
                                  nan_treatment=nan_treatment, mask=mask,
                                  fill_value=np.nan)
    kernel = _copy_input_if_needed(kernel, dtype=complex, order='C',
                                   nan_treatment=None, mask=None,
                                   fill_value=0)

    # Check that the number of dimensions is compatible
    if array.ndim != kernel.ndim:
        raise ValueError("Image and kernel must have same number of "
                         "dimensions")

    arrayshape = array.shape
    kernshape = kernel.shape

    array_size_B = (np.product(arrayshape, dtype=np.int64) *
                    np.dtype(complex_dtype).itemsize)*u.byte
    if array_size_B > 1*u.GB and not allow_huge:
        raise ValueError("Size Error: Arrays will be {}.  Use "
                         "allow_huge=True to override this exception."
                         .format(human_file_size(array_size_B.to_value(u.byte))))

    # NaN and inf catching
    nanmaskarray = np.isnan(array) | np.isinf(array)
    if nan_treatment == 'fill':
        array[nanmaskarray] = fill_value
    else:
        array[nanmaskarray] = 0
    nanmaskkernel = np.isnan(kernel) | np.isinf(kernel)
    kernel[nanmaskkernel] = 0

    if normalize_kernel is True:
        if kernel.sum() < 1. / MAX_NORMALIZATION:
            raise Exception("The kernel can't be normalized, because its sum is "
                            "close to zero. The sum of the given kernel is < {}"
                            .format(1. / MAX_NORMALIZATION))
        kernel_scale = kernel.sum()
        normalized_kernel = kernel / kernel_scale
        kernel_scale = 1  # if we want to normalize it, leave it normed!
    elif normalize_kernel:
        # try this.  If a function is not passed, the code will just crash... I
        # think type checking would be better but PEPs say otherwise...
        kernel_scale = normalize_kernel(kernel)
        normalized_kernel = kernel / kernel_scale
    else:
        kernel_scale = kernel.sum()
        if np.abs(kernel_scale) < normalization_zero_tol:
            if nan_treatment == 'interpolate':
                raise ValueError('Cannot interpolate NaNs with an unnormalizable kernel')
            else:
                # the kernel's sum is near-zero, so it can't be scaled
                kernel_scale = 1
                normalized_kernel = kernel
        else:
            # the kernel is normalizable; we'll temporarily normalize it
            # now and undo the normalization later.
            normalized_kernel = kernel / kernel_scale

    if boundary is None:
        warnings.warn("The convolve_fft version of boundary=None is "
                      "equivalent to the convolve boundary='fill'.  There is "
                      "no FFT equivalent to convolve's "
                      "zero-if-kernel-leaves-boundary", AstropyUserWarning)
        if psf_pad is None:
            psf_pad = True
        if fft_pad is None:
            fft_pad = True
    elif boundary == 'fill':
        # create a boundary region at least as large as the kernel
        if psf_pad is False:
            warnings.warn("psf_pad was set to {}, which overrides the "
                          "boundary='fill' setting.".format(psf_pad),
                          AstropyUserWarning)
        else:
            psf_pad = True
        if fft_pad is None:
            # default is 'True' according to the docstring
            fft_pad = True
    elif boundary == 'wrap':
        if psf_pad:
            raise ValueError("With boundary='wrap', psf_pad cannot be enabled.")
        psf_pad = False
        if fft_pad:
            raise ValueError("With boundary='wrap', fft_pad cannot be enabled.")
        fft_pad = False
        fill_value = 0  # force zero; it should not be used
    elif boundary == 'extend':
        raise NotImplementedError("The 'extend' option is not implemented "
                                  "for fft-based convolution")

    # find ideal size (power of 2) for fft.
    # Can add shapes because they are tuples
    if fft_pad:  # default=True
        if psf_pad:  # default=False
            # add the dimensions and then take the max (bigger)
            fsize = 2 ** np.ceil(np.log2(
                np.max(np.array(arrayshape) + np.array(kernshape))))
        else:
            # add the shape lists (max of a list of length 4) (smaller)
            # also makes the shapes square
            fsize = 2 ** np.ceil(np.log2(np.max(arrayshape + kernshape)))
        newshape = np.full((array.ndim, ), fsize, dtype=int)
    else:
        if psf_pad:
            # just add the biggest dimensions
            newshape = np.array(arrayshape) + np.array(kernshape)
        else:
            newshape = np.array([np.max([imsh, kernsh])
                                 for imsh, kernsh in zip(arrayshape, kernshape)])

    # perform a second check after padding
    array_size_C = (np.product(newshape, dtype=np.int64) *
                    np.dtype(complex_dtype).itemsize)*u.byte
    if array_size_C > 1*u.GB and not allow_huge:
        raise ValueError("Size Error: Arrays will be {}.  Use "
                         "allow_huge=True to override this exception."
                         .format(human_file_size(array_size_C)))

    # separate each dimension by the padding size...  this is to determine the
    # appropriate slice size to get back to the input dimensions
    arrayslices = []
    kernslices = []
    for ii, (newdimsize, arraydimsize, kerndimsize) in enumerate(zip(newshape, arrayshape, kernshape)):
        center = newdimsize - (newdimsize + 1) // 2
        arrayslices += [slice(center - arraydimsize // 2,
                              center + (arraydimsize + 1) // 2)]
        kernslices += [slice(center - kerndimsize // 2,
                             center + (kerndimsize + 1) // 2)]
    arrayslices = tuple(arrayslices)
    kernslices = tuple(kernslices)

    if not np.all(newshape == arrayshape):
        if np.isfinite(fill_value):
            bigarray = np.ones(newshape, dtype=complex_dtype) * fill_value
        else:
            bigarray = np.zeros(newshape, dtype=complex_dtype)
        bigarray[arrayslices] = array
    else:
        bigarray = array

    if not np.all(newshape == kernshape):
        bigkernel = np.zeros(newshape, dtype=complex_dtype)
        bigkernel[kernslices] = normalized_kernel
    else:
        bigkernel = normalized_kernel

    arrayfft = fftn(bigarray)
    # need to shift the kernel so that, e.g., [0,0,1,0] -> [1,0,0,0] = unity
    if kernfft is None:
        kernfft = fftn(np.fft.ifftshift(bigkernel))
    else:
        if not kernfft.shape == arrayfft.shape:
            raise RuntimeError(f'kernfft.shape={kernfft.shape} != arrayfft.shape={arrayfft.shape}')
    if return_kernfft:
        return kernfft
       
    fftmult = arrayfft * kernfft
    
    interpolate_nan = (nan_treatment == 'interpolate')
    if interpolate_nan:
        if not np.isfinite(fill_value):
            bigimwt = np.zeros(newshape, dtype=complex_dtype)
        else:
            bigimwt = np.ones(newshape, dtype=complex_dtype)

        bigimwt[arrayslices] = 1.0 - nanmaskarray * interpolate_nan
        wtfft = fftn(bigimwt)

        # You can only get to this point if kernel_is_normalized
        wtfftmult = wtfft * kernfft
        wtsm = ifftn(wtfftmult)
        # need to re-zero weights outside of the image (if it is padded, we
        # still don't weight those regions)
        bigimwt[arrayslices] = wtsm.real[arrayslices]
        
    if np.isnan(fftmult).any():
        # this check should be unnecessary; call it an insanity check
        raise ValueError("Encountered NaNs in convolve.  This is disallowed.")

    fftmult *= kernel_scale

    if return_fft:
        return fftmult

    if interpolate_nan:
        with np.errstate(divide='ignore', invalid='ignore'):
            # divide by zeros are expected here; if the weight is zero, we want
            # the output to be nan or inf
            rifft = (ifftn(fftmult))
            if interpolate_nan:
                rifft /= bigimwt
        if interpolate_nan:
            if min_wt > 0.:
                rifft[bigimwt < min_wt] = np.nan
            else:
                # Set anything with no weight to zero (taking into account
                # slight offsets due to floating-point errors).
                rifft[bigimwt < 10 * np.finfo(bigimwt.dtype).eps] = 0.0
    else:
        rifft = ifftn(fftmult)

    if preserve_nan:
        rifft[arrayslices][nanmaskarray] = np.nan

    if crop:
        result = rifft[arrayslices].real
        return result
    else:
        return rifft.real


# Using fftw in scarlet; diffs with scarlet/fft.py:
# 
#     3c3
#     < #import autograd.numpy as np
#     ---
#     > import autograd.numpy as np
#     169c178
#     <         image = irfftw2(image_fft, s=fft_shape, axes=axes)
#     ---
#     >         image = np.fft.irfftn(image_fft, fft_shape, axes=axes)
#     206c215
#     <             self._fft[fft_key] = rfftw2(np.fft.ifftshift(image, axes), axes=axes)
#     ---
#     >             self._fft[fft_key] = np.fft.rfftn(np.fft.ifftshift(image, axes), axes=axes)
#     288c299
#     < def convolve_fftw(image1, image2, padding=3, axes=(-2, -1)):
#     ---
#     > def convolve(image1, image2, padding=3, axes=(-2, -1)):
# 

# In[6]:


# 

import operator

#import autograd.numpy as np
from scipy import fftpack

def _centered(arr, newshape):
    """Return the center newshape portion of the array.
    This function is used by `fft_convolve` to remove
    the zero padded region of the convolution.
    Note: If the array shape is odd and the target is even,
    the center of `arr` is shifted to the center-right
    pixel position.
    This is slightly different than the scipy implementation,
    which uses the center-left pixel for the array center.
    The reason for the difference is that we have
    adopted the convention of `np.fft.fftshift` in order
    to make sure that changing back and forth from
    fft standard order (0 frequency and position is
    in the bottom left) to 0 position in the center.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)

    if not np.all(newshape <= currshape):
        msg = (
            "arr must be larger than newshape in both dimensions, received {0}, and {1}"
        )
        raise ValueError(msg.format(arr.shape, newshape))

    startind = (currshape - newshape + 1) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]


def _pad(arr, newshape, axes=None):
    """Pad an array to fit into newshape
    Pad `arr` with zeros to fit into newshape,
    which uses the `np.fft.fftshift` convention of moving
    the center pixel of `arr` (if `arr.shape` is odd) to
    the center-right pixel in an even shaped `newshape`.
    """
    if axes is None:
        newshape = np.asarray(newshape)
        currshape = np.array(arr.shape)
        dS = newshape - currshape
        startind = (dS + 1) // 2
        endind = dS - startind
        pad_width = list(zip(startind, endind))
    else:
        # only pad the axes that will be transformed
        pad_width = [(0, 0) for axis in arr.shape]
        try:
            len(axes)
        except TypeError:
            axes = [axes]
        for a, axis in enumerate(axes):
            dS = newshape[a] - arr.shape[axis]
            startind = (dS + 1) // 2
            endind = dS - startind
            pad_width[axis] = (startind, endind)
    return np.pad(arr, pad_width, mode="constant")


def _get_fft_shape(img1, img2, padding=3, axes=None, max=False):
    """Return the fast fft shapes for each spatial axis
    Calculate the fast fft shape for each dimension in
    axes.
    """
    shape1 = np.asarray(img1.shape)
    shape2 = np.asarray(img2.shape)
    # Make sure the shapes are the same size
    if len(shape1) != len(shape2):
        msg = (
            "img1 and img2 must have the same number of dimensions, but got {0} and {1}"
        )
        raise ValueError(msg.format(len(shape1), len(shape2)))
    # Set the combined shape based on the total dimensions
    if axes is None:
        if max:
            shape = np.max([shape1, shape2], axis=1)
        else:
            shape = shape1 + shape2
    else:
        shape = np.zeros(len(axes), dtype='int')
        try:
            len(axes)
        except TypeError:
            axes = [axes]
        for n, ax in enumerate(axes):
            shape[n] = shape1[ax] + shape2[ax]
            if max == True:
                shape[n] = np.max([shape1[ax], shape2[ax]])

    shape += padding
    # Use the next fastest shape in each dimension
    shape = [fftpack.helper.next_fast_len(s) for s in shape]
    # autograd.numpy.fft does not currently work
    # if the last dimension is odd
    while shape[-1] % 2 != 0:
        shape[-1] += 1
        shape[-1] = fftpack.helper.next_fast_len(shape[-1])

    return shape


class Fourier(object):
    """An array that stores its Fourier Transform
    The `Fourier` class is used for images that will make
    use of their Fourier Transform multiple times.
    In order to prevent numerical artifacts the same image
    convolved with different images might require different
    padding, so the FFT for each different shape is stored
    in a dictionary.
    """

    def __init__(self, image, image_fft=None):
        """Initialize the object
        Parameters
        ----------
        image: array
            The real space image.
        image_fft: dict
            A dictionary of {shape: fft_value} for which each different
            shape has a precalculated FFT.
        axes: int or tuple
            The dimension(s) of the array that will be transformed.
        """
        if image_fft is None:
            self._fft = {}
        else:
            self._fft = image_fft
        self._image = image

    @staticmethod
    def from_fft(image_fft, fft_shape, image_shape, axes=None):
        """Generate a new Fourier object from an FFT dictionary
        If the fft of an image has been generated but not its
        real space image (for example when creating a convolution kernel),
        this method can be called to create a new `Fourier` instance
        from the k-space representation.
        Parameters
        ----------
        image_fft: array
            The FFT of the image.
        fft_shape: tuple
            "Fast" shape of the image used to generate the FFT.
            This will be different than `image_fft.shape` if
            any of the dimensions are odd, since `np.fft.rfft`
            requires an even number of dimensions (for symmetry),
            so this tells `np.fft.irfft` how to go from
            complex k-space to real space.
        image_shape: tuple
            The shape of the image *before padding*.
            This will regenerate the image with the extra
            padding stripped.
        axes: int or tuple
            The dimension(s) of the array that will be transformed.
        Returns
        -------
        result: `Fourier`
            A `Fourier` object generated from the FFT.
        """
        if axes is None:
            axes = range(len(image_fft))
        all_axes = range(len(image_shape))
        image = irfftw2(image_fft, s=fft_shape, axes=axes)
        # Shift the center of the image from the bottom left to the center
        image = np.fft.fftshift(image, axes=axes)
        # Trim the image to remove the padding added
        # to reduce fft artifacts
        image = _centered(image, image_shape)
        key = (tuple(fft_shape), tuple(axes), tuple(all_axes))

        return Fourier(image, {key: image_fft})

    @property
    def image(self):
        """The real space image"""
        return self._image

    @property
    def shape(self):
        """The shape of the real space image"""
        return self.image.shape

    def fft(self, fft_shape, axes):
        """The FFT of an image for a given `fft_shape` along desired `axes`
        """
        try:
            iter(axes)
        except TypeError:
            axes = (axes,)
        all_axes = range(len(self.image.shape))
        fft_key = (tuple(fft_shape), tuple(axes), tuple(all_axes))

        # If this is the first time calling `fft` for this shape,
        # generate the FFT.
        if fft_key not in self._fft:
            if len(fft_shape) != len(axes):
                msg = "fft_shape self.axes must have the same number of dimensions, got {0}, {1}"
                raise ValueError(msg.format(fft_shape, axes))
            image = _pad(self.image, fft_shape, axes)
            self._fft[fft_key] = rfftw2(np.fft.ifftshift(image, axes), axes=axes)
        return self._fft[fft_key]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        # Make the index a tuple
        if not hasattr(index, "__getitem__"):
            index = tuple([index])

        # Axes that are removed from the shape of the new object
        removed = np.array(
            [
                n
                for n, idx in enumerate(index)
                if not isinstance(idx, slice) and idx is not None
            ]
        )

        # Create views into the fft transformed values, appropriately adjusting
        # the shapes for the new axes

        fft_kernels = {
            (
                tuple(
                    [s for idx, s in enumerate(key[0]) if key[1][idx] not in removed]
                ),
                tuple(
                    [a for ida, a in enumerate(key[1]) if key[1][ida] not in removed]
                ),
                tuple(
                    [
                        aa
                        for idaa, aa in enumerate(key[2])
                        if key[2][idaa] not in removed
                    ]
                ),
            ): kernel[index]
            for key, kernel in self._fft.items()
        }
        return Fourier(self.image[index], fft_kernels)


def _kspace_operation(image1, image2, padding, op, shape, axes):
    """Combine two images in k-space using a given `operator`
    `image1` and `image2` are required to be `Fourier` objects and
    `op` should be an operator (either `operator.mul` for a convolution
    or `operator.truediv` for deconvolution). `shape` is the shape of the
    output image (`Fourier` instance).
    """
    if len(image1.shape) != len(image2.shape):
        msg = "Both images must have the same number of axes, got {0} and {1}"
        raise Exception(msg.format(len(image1.shape), len(image2.shape)))
    fft_shape = _get_fft_shape(image1.image, image2.image, padding, axes)
    convolved_fft = op(image1.fft(fft_shape, axes), image2.fft(fft_shape, axes))
    # why is shape not image1.shape? images are never padded
    convolved = Fourier.from_fft(convolved_fft, fft_shape, shape, axes)
    return convolved


def match_psfs(psf1, psf2, padding=3, axes=(-2, -1)):
    """Calculate the difference kernel between two psfs
    Parameters
    ----------
    psf1: `Fourier`
        `Fourier` object representing the psf and it's FFT.
    psf2: `Fourier`
        `Fourier` object representing the psf and it's FFT.
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    axes: tuple or None
        Axes that contain the spatial information for the PSFs.
    """
    if psf1.shape[0] < psf2.shape[0]:
        shape = psf2.shape
    else:
        shape = psf1.shape
    return _kspace_operation(psf1, psf2, padding, operator.truediv, shape, axes=axes)


def convolve_fftw(image1, image2, padding=3, axes=(-2, -1)):
    """Convolve two images
    Parameters
    ----------
    image1: `Fourier`
        `Fourier` object represeting the image and it's FFT.
    image2: `Fourier`
        `Fourier` object represeting the image and it's FFT.
    padding: int
        Additional padding to use when generating the FFT
        to supress artifacts.
    """
    return _kspace_operation(
        image1, image2, padding, operator.mul, image1.shape, axes=axes
    )


# In[7]:


fwhm = 3
modelpsf = gs.Gaussian(fwhm=fwhm)

sizes_psf = (41,)# 51)
sizes_img = np.arange(20, 250, 1)

sizes_psf_plot = set(sizes_psf[:1])
sizes_img_plot = set(sizes_img[:1])

# Set to test for very slightly non-square images; set to zero for squares
shrink_y = 2
gsp_rough = gs.GSParams(xvalue_accuracy=1e-3)

results = {}

if mpf_draw:
    def mpf_draw(model):
        model.evaluate(get_likelihood=False, keep_images=True)
        return model.data.exposures[''][0].meta['img_model']

if pyfftw_all:
    def rfftw2(*args, **kwargs):
        rfft2 = pyfftw.builders.rfft2(*args, **kwargs)
        return rfft2(*args)
    
    def irfftw2(*args, **kwargs):
        irfft2 = pyfftw.builders.irfft2(*args, **kwargs)
        return irfft2(*args)

    def fftw2(*args, **kwargs):
        fftw2 = pyfftw.builders.fft2(*args, **kwargs)
        return fftw2(*args, **kwargs)
    
    def ifftw2(*args, **kwargs):
        ifft2 = pyfftw.builders.ifft2(*args, **kwargs)
        return ifft2(*args, **kwargs)
    
for size_psf in sizes_psf:
    psf_gs = modelpsf.drawImage(nx=size_psf, ny=size_psf, scale=1, method='no_pixel', dtype=np.float64)
    psf = psf_gs.array
    psf_gs = gs.InterpolatedImage(psf_gs)
    plot_psf = size_psf in sizes_psf_plot
    results_psf = {}
    
    for size_img in sizes_img:
        print(f'PSF {size_psf} x image {size_img}')
        plot = plot_psf and (size_img in sizes_img_plot)
        size_img_y = size_img - shrink_y
        axrat = 0.5
        galaxy = gs.Sersic(half_light_radius=size_img/10, n=1).shear(
            q=axrat, beta=gs.Angle(45, unit=gs.degrees))
        r_eff = size_img/10
        galaxy_rough = gs.Sersic(half_light_radius=r_eff, n=1, gsparams=gsp_rough).shear(
            q=0.5, beta=gs.Angle(45, unit=gs.degrees))
        img = galaxy.drawImage(nx=size_img, ny=size_img_y, scale=1, method='real_space', dtype=np.float64).array
        psf_fft = convolve_fft(img, psf, normalize_kernel=False, fft_pad=False, psf_pad=True, nan_treatment='fill', return_kernfft=True)
        
        print(f'Sum PSF={np.sum(psf):.3e} img={np.sum(img):.3e}')
        
        if scarlet_direct:
            coords = get_filter_coords(psf)
            bounds = get_filter_bounds(coords.reshape(-1, 2))
        if scarlet_fft:
            img_scarletfft = fft.Fourier(img)
            psf_scarletfft = fft.Fourier(psf)
        
        args_profit = (img, psf, size_img, size_img_y, size_psf, size_psf,)
        methods = {
            # fft_pad claims to pad the image to the nearest 2^n, but doesn't actually seem to do it here
            # we leave it to its default true here
            'astropy.np.fft': lambda: convolve_fft(img, psf, normalize_kernel=False, kernfft=psf_fft, fft_pad=False, psf_pad=True, nan_treatment='fill'),
            'astropy.np.fftc': lambda: convolve_fft(img, psf, normalize_kernel=False, kernfft=psf_fft, fft_pad=False, psf_pad=True, nan_treatment='fill'),
            'signal.direct': lambda: signal.convolve(img, psf, method="direct", mode="same"),
            'signal.fft': lambda: signal.convolve(img, psf, method="fft", mode="same"),
            # This takes ages and ages so don't run it unless you're patient or using small boxes
            #'galsim.draw-d': lambda: gs.Convolve(
            #    galaxy, psf_gs, real_space=True, gsparams=gsp_rough).drawImage(
            #        nx=size_img, ny=size_img_y, method='no_pixel', dtype=np.float64).array,
            'galsim.draw-f': lambda: gs.Convolve(galaxy, modelpsf, real_space=False).drawImage(
                nx=size_img, ny=size_img_y, method='fft', dtype=np.float64).array,
        }
        if cv2_all:
            methods['cv2'] = lambda: cv2.filter2D(img, -1, psf, borderType=cv2.BORDER_CONSTANT)
        if profit_all:
            # Setup pyprofit convolvers
            convolvers = {
                f'{c_type}{"-reuse" if reuse else ""}': profit.make_convolver(
                    width=size_img, height=size_img_y, psf=psf, convolver_type=c_type, reuse_psf_fft=reuse, fft_effort=2)
                for c_type, reuses in {
                    'brute': (False,),
                    'brute-old': (False,),
                    'fft': (False, True),
                }.items()
                for reuse in reuses
            }
            methods['profit.direct'] = lambda: np.array(profit.convolve(convolvers['brute'], *args_profit)[0])
            methods['profit.direct-old'] = lambda: np.array(profit.convolve(convolvers['brute-old'], *args_profit)[0])
            methods['profit.fft'] = lambda: np.array(profit.convolve(convolvers['fft-reuse'], *args_profit)[0])
        if pyfftw_all:
            # fft_pad claims to pad the image to the nearest 2^n which is not necessarily beneficial
            methods['astropy.fftw3'] = lambda: convolve_fft(img, psf, fftn=fftw2, ifftn=ifftw2, normalize_kernel=False, kernfft=psf_fft, fft_pad=False, psf_pad=True, nan_treatment='fill')
            methods['scarlet.fftw3'] = lambda: convolve_fftw(img_scarletfft, psf_scarletfft).image

        if scarlet_direct:
            methods['scarlet.direct'] = lambda: convolve(img[None, :, :], psf[None, :, :], bounds)[0, :, :]
        if scarlet_fft:
            methods['scarlet.fft'] = lambda: fft.convolve(img_scarletfft, psf_scarletfft).image
        
        if mpf_draw:
            size_mpf = r_eff
            galaxy_mpf = mpfFit.get_model(
                {'': 1}, "mgsersic8:1", (size_img, size_img_y),
                sigma_xs=[size_mpf], sigma_ys=[size_mpf], rhos=[0.6], slopes=[1],
                engine='galsim',
                engineopts={'use_fast_gauss': True, 'drawmethod': mpfObj.draw_method_pixel['galsim']},
            )
            psf_mpf = mpfFit.get_model(
                {'': 1}, "gaussian:1", (0, 0),
                sigma_xs=[fwhm/2.], sigma_ys=[fwhm/2.], rhos=[0.],
                engine='galsim',
                engineopts={'use_fast_gauss': True, 'drawmethod': mpfObj.draw_method_pixel['galsim']},
            ).sources[0]
            galaxy_mpf.data.exposures[''][0].image = mpfFit.ImageEmpty((size_img_y, size_img))
            galaxy_mpf.data.exposures[''][0].psf = mpfObj.PSF(band='', engine='galsim', model=psf_mpf, is_model_pixelated=True)
            methods['multiprofit.draw-d'] = lambda: mpf_draw(galaxy_mpf)
        
        method_ref = 'signal.direct'
        conv_ref = methods[method_ref]()
        conv_ref_log = np.log10(conv_ref)
        if plot:
            imshow((np.log10(img), np.log10(psf), conv_ref_log))
        
        for name, func in methods.items():
            if name not in results_psf:
                results_psf[name] = []
            str_func = " | ".join(x.strip() for x in inspect.getsourcelines(func)[0])
            print(f'Running {name}: {{{str_func}}}... ', end='', flush=True)
            # galsim.real is painfully slow
            repeat = 1 if (name == 'galsim.draw.real' or name == 'signal.direct') else 16
            number = repeat
            n_eval = number*repeat
            msg_time = f'timing {n_eval}x'

            if name == method_ref:
                diff = 0
                str_diff = ''
                print(msg_time)
            else:
                time = timer()
                conv = func()
                print(f' took {timer()-time:.3e} for plot eval; {msg_time}')
                if plot:
                    imshow((conv, conv_ref, conv-conv_ref, np.log10(conv) - conv_ref_log), title=name)
                diff = np.sum(np.abs((conv - conv_ref)))
                str_diff = f', abs. diff.: {diff:.3e}'
            time_run = timer()
            result = np.min(timeit.repeat(func, number=repeat, repeat=repeat))/repeat
            time_run = timer() - time_run
            print(f'{name} {result:.3e}s{str_diff}; Ran {n_eval}x in {time_run:.3e}s', flush=True)
            results_psf[name].append((result, size_img, diff,))
    results[size_psf] = results_psf


# In[8]:


for size_psf, results_psf in results.items():
    plt.figure()
    for name, values in results_psf.items():
        n_values = len(values)
        size_img, time = np.zeros(n_values), np.zeros(n_values)
        for idx in range(n_values):
            value = values[idx]
            size_img[idx] = value[1]
            time[idx] = value[0]
        plt.plot(size_img, time, label=name, linestyle='-' if 'fft' in name else ('--' if 'direct' in name else '-.'))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('image side length (px)')
    plt.ylabel('time (s)')
    plt.title(f'{size_psf}^2 PSF')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

