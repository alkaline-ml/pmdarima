from matplotlib.testing.exceptions import ImageComparisonFailure
import matplotlib._png as _png

import os
import numpy as np

def calculate_rms(expected_image, actual_image):
    """

    Calculate the per-pixel errors, then compute the root mean square error.

    Parameters
    ----------
    expected_image : str
        Singed integer representation of expected image
    actual_image : str
        Signed integer representation of actual/generated image.

    Returns
    -------
    rms: float
        RMSE of the the two images.


    References
    ----------
    .. [1] https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/testing/compare.py
    """


    if expected_image.shape != actual_image.shape:
        raise ImageComparisonFailure(
            "Image sizes do not match expected size: {} "
            "actual size {}".format(expected_image.shape, actual_image.shape))
    # Convert to float to avoid overflowing finite integer types.
    return np.sqrt(((expected_image - actual_image).astype(float) ** 2).mean())


def compare_images(expected, actual, tol):
    """
    Compare two "image" files checking differences within a tolerance.
    The two given filenames may point to files which are convertible to
    PNG via the `.converter` dictionary. The underlying RMS is calculated
    with the `.calculate_rms` function.
    Parameters
    ----------
    expected : str
        The filename of the expected image.
    actual : str
        The filename of the actual image.
    tol : float
        The tolerance (a color value difference, where 255 is the
        maximal difference).  The test fails if the average pixel
        difference is greater than this value.
    in_decorator : bool
        Determines the output format. If called from image_comparison
        decorator, this should be True. (default=False)
    Returns
    -------
    comparison_result : None or dict or str
        Return *None* if the images are equal within the given tolerance.
        If the images differ, the return value depends on  *in_decorator*.
        If *in_decorator* is true, a dict with the following entries is
        returned:
        - *rms*: The RMS of the image difference.
        - *expected*: The filename of the expected image.
        - *actual*: The filename of the actual image.
        - *diff_image*: The filename of the difference image.
        - *tol*: The comparison tolerance.
        Otherwise, a human-readable multi-line string representation of this
        information is returned.
    Examples
    --------
    ::
        img1 = "./baseline/plot.png"
        img2 = "./output/plot.png"
        compare_images(img1, img2, 0.001)

    References
    ----------
    .. [1] https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/testing/compare.py

    """
    if not os.path.exists(actual):
        raise Exception("Output image %s does not exist." % actual)

    if os.stat(actual).st_size == 0:
        raise Exception("Output image file %s is empty." % actual)

    if not os.path.exists(expected):
        raise IOError('Baseline image %r does not exist.' % expected)

    # open the image files and remove the alpha channel (if it exists)
    expected_image = _png.read_png_int(expected)
    actual_image = _png.read_png_int(actual)
    expected_image = expected_image[:, :, :3]
    actual_image = actual_image[:, :, :3]

    if tol <= 0:
        if np.array_equal(expected_image, actual_image):
            return None

    # convert to signed integers, so that the images can be subtracted without
    # overflow
    expected_image = expected_image.astype(np.int16)
    actual_image = actual_image.astype(np.int16)

    rms = calculate_rms(expected_image, actual_image)
    if rms <= tol:
        return None
    else:
        results = dict(rms=rms, expected=str(expected),
                       actual=str(actual), tol=tol)

        # Then the results should be a string suitable for stdout.
        template = ['Error: Image files did not match.',
                    'RMS Value: {rms}',
                    'Expected:  \n    {expected}',
                    'Actual:    \n    {actual}',
                    'Tolerance: \n    {tol}', ]
        results = '\n  '.join([line.format(**results) for line in template])
        raise AssertionError(results)
