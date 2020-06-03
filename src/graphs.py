"""
In this file, we define functions for plotting images from
the dataset using matplotlib.pyplot.
"""
from matplotlib import pyplot as plt
import seaborn as sns
from utils import get_label_name
import numpy as np
from typing import Union
import matplotlib.colors as colors


def img_for_show(raw_img: np.ndarray) -> np.ndarray:
    """
    Creates new image prepared to show from raw data from batch.

    :param raw_img: 1D format of image
    :return: 3D format (y, x, rgb) ready for pyplot's imshow
    """
    return raw_img.reshape((32, 32, 3), order='F').swapaxes(0, 1)


def plot_raw_img(raw_img: np.ndarray, label: Union[None, int], ax,
                 fontsize: str = 'medium') -> None:
    """
    Plots single image on specified ax (see Axes).

    :param raw_img: format as in batch (1D array [R, G, B])
    :param label: integer label (None for no title)
    :param ax: ax to plot on
    :param fontsize: passed to ax.set_title
    """
    ax.imshow(img_for_show(raw_img))
    if label is not None:
        ax.set_title(f"{label}: {get_label_name(label)}", fontsize=fontsize)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


def plot_rgb_hist(raw_imgs: np.ndarray, ax, **kwargs) -> None:
    """
    Plots histograms of image(s)'s R, G, B amount using seaborn

    :param raw_imgs: array of images (1D format) or just one such image
    :param ax: ax on which to plot
    :param kwargs: passed to seaborn.distplot
    """
    r = raw_imgs.reshape((-1, 3072))[:, :1024].reshape(-1)
    g = raw_imgs.reshape((-1, 3072))[:, 1024:2048].reshape(-1)
    b = raw_imgs.reshape((-1, 3072))[:, 2048:].reshape(-1)
    sns.distplot(r, color='red', ax=ax, **kwargs)
    sns.distplot(g, color='green', ax=ax, **kwargs)
    sns.distplot(b, color='blue', ax=ax, **kwargs)


def plot_random(imgs: np.ndarray, labels: np.ndarray,
                nrows: int = 1, ncols: int = 1,
                fontsize: str = 'medium', **kwargs) -> None:
    """
    Plots random images from batch using matplotlib pyplot.
    User should typically call pyplot.show() after this function.
    Note: Number of plotted images is nrows * ncols.

    :param imgs: images in form 1D array [Rvalue, Gvalues, Bvalues]
    :param labels: labels (numerical) attached to images. If all labels are the
                   same, only array of one element can be passed
    :param nrows: number of rows of images
    :param ncols: number of cols of images
    :param fontsize: passed to pyplot.Axes.set_title
    :param kwargs: passed to pyplot.subplots
    """
    if imgs.shape[0] < nrows * ncols:
        # If there is enough to create a full row, go for it
        if imgs.shape[0] >= ncols:
            nrows = int(np.ceil(imgs.shape[0] / ncols))
        else:
            nrows = 1
            ncols = imgs.shape[0]
    imgs_count = min(nrows * ncols, imgs.shape[0])

    fig, axs = plt.subplots(nrows, ncols, **kwargs)
    axs_flat = np.reshape(axs, -1)
    random = np.random.choice(imgs.shape[0], imgs_count, replace=False)
    one_label = (len(labels) == 1)

    for i in range(imgs_count):
        img = imgs[random[i]]
        label = labels[random[i]] if not one_label else labels[0]
        plot_raw_img(img, label, axs_flat[i], fontsize=fontsize)
        
        
def plot_images(imgs: np.ndarray) -> None:
    """
    Plots all images given, using matplotlib.pyplot in single row.
    
    :param imgs: images in form 1D array [Rvalue, Gvalues, Bvalues]
    """
    width = len(imgs)
    height = 1
    
    fig, axs = plt.subplots(height, width)
    counter = 0
    for ax in np.reshape(axs, -1):
        plot_raw_img(imgs[counter], label=None, ax=ax)
        counter += 1

        
def hue_saturation_histogram(img, ax):
    # normalise picture
    img = img_for_show(img)
    img = img.astype(float) / 255.0
    
    # convert ho hsv
    img_hsv = colors.rgb_to_hsv(img[...,:3])
    
    # hue
    lu1 = img_hsv[..., 0].flatten()
    ax.hist(lu1 * 360, bins=360, range=(0.0, 360.0), 
             histtype='stepfilled', color='r', label='Hue')
    # saturation
    lu2=img_hsv[..., 1].flatten()
    ax.hist(lu2, bins=100, range=(0.0, 1.0), 
             histtype='stepfilled', color='g', label='Saturation')
   
    # intensity
    lu3=img_hsv[..., 2].flatten()
    ax.hist(lu3 * 255, bins=256, range=(0.0, 255.0),
             histtype='stepfilled', color='b', label='Intesity')
