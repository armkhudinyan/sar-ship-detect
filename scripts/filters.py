from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


def lee_filter(img, size):
    '''
    Applies Lee Filter

    Parameters
    ----------
    img : array-like, image to apply filter on
    size : integer,
           The sizes of the uniform filter are given for each axis
    Returns
    -------
    Filtered array. Has the same shape as `input`.
    '''
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output