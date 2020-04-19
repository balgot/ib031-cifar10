from matplotlib import pyplot as plt
import seaborn as sns
from utils import get_label_name
import numpy as np
from typing import Union


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


def plot_random(imgs: np.ndarray, labels: np.ndarray, nrows: int = 1, ncols: int = 1,
                fontsize: str = 'medium', **kwargs) -> None:
    """
    Plots random images from batch using matplotlib pyplot.
    User should typically call pyplot.show() after this function.
    Note: Number of plotted images is nrows * ncols.

    :param imgs: images in form 1D array [Rvalue, Gvalues, Bvalues]
    :param labels: labels (numerical) attached to images
    :param nrows: number of rows of images
    :param ncols: number of cols of images
    :param fontsize: passed to pyplot.Axes.set_title
    :param kwargs: passed to pyplot.subplots
    """
    fig, axs = plt.subplots(nrows, ncols, **kwargs)

    for ax in axs.reshape(-1):
        random = np.random.randint(0, imgs.shape[0])
        img = imgs[random]
        label = labels[random]
        plot_raw_img(img, label, ax, fontsize=fontsize)
        
        
def plot_images(imgs: np.ndarray) -> None:
    """
    Plots all images given, using matplotlib.pyplot in single row.
    
    :param imgs: images in form 1D array [Rvalue, Gvalues, Bvalues]
    """
    width = len(imgs)
    height = 1
    
    fig, axs = plt.subplots(height, width)
    counter = 0
    for ax in axs.reshape(-1):
        plot_raw_img(imgs[counter], label=None, ax=ax)
        counter += 1