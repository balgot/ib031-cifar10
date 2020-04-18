from matplotlib import pyplot as plt
import numpy as np
import utils
import graphs
from typing import Dict


def plot_avg_imgs(batch: Dict, with_histogram: bool = True) -> None:
    """
    Plot average image and optionally also histogram of RGB values for each
    category.

    :param batch: dictionary with images and labels
    :param with_histogram: whether to plot also histograms
    """
    nrows = 4 if with_histogram else 2
    fig, axs = plt.subplots(nrows, 5, figsize=(15, 10))
    hist_kws = {'range': (50, 200)}

    for i in range(10):
        imgs = utils.imgs_of_cat(batch, i)
        avg_img = np.mean(imgs, axis=0)
        x, y = i % 5, i // 5
        if with_histogram:
            y *= 2

        graphs.plot_raw_img(avg_img.astype('int'), i, axs[y][x])
        if with_histogram:
            graphs.plot_rgb_hist(avg_img, axs[y + 1][x], hist_kws=hist_kws)


def plot_global_hist(batch: Dict, sample_size: int = 50) -> None:
    """
    Histogram of RGB values from whole batch. Take all values into account
    instead of making an average image. But because whole dataset is too large,
    we use only random subsample of default size 50 images from each category.

    :param batch: dictionary with images and labels
    :param sample_size: size of random sample of each category
    """
    fig, axs = plt.subplots(2, 5, figsize=(15, 8))

    for i in range(10):
        imgs = utils.imgs_of_cat(batch, i)
        imgs = imgs[np.random.choice(imgs.shape[0], sample_size)]
        x, y = i % 5, i // 5

        axs[y][x].set_title(utils.get_label_name(i))
        graphs.plot_rgb_hist(imgs, axs[y][x])


if __name__ == '__main__':
    batch = utils.load_data_batch(1)

    # Print dataset size info about each category
    for i in range(10):
        imgs = utils.imgs_of_cat(batch, i)
        print(f"Category {i}: {utils.get_label_name(i)}")
        print(f"Size: {imgs.shape[0]}")
        print()

    # Find average image and plot histogram of RGB values for each category.
    plot_avg_imgs(batch)
    plt.tight_layout()
    plt.show()

    # Lets look at global histogram of each category. Use default size 50.
    plot_global_hist(batch)
    plt.tight_layout()
    plt.show()
