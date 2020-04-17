from matplotlib import pyplot as plt
import numpy as np
import utils


if __name__ == '__main__':
    batch = utils.load_data_batch(1)

    """
    Print dataset size info about each category
    """
    for i in range(10):
        imgs = utils.imgs_of_cat(batch, i)
        print("Category {}: {}".format(i, utils.get_lname(i)))
        print("Size: {}".format(imgs.shape[0]))
        print()

    """
    Find average image and plot histogram of RGB values for each category
    """

    fig, axs = plt.subplots(4, 5, figsize=(15, 10))
    hist_kws = { 'range' : (50, 200) }

    for i in range(10):
        imgs = utils.imgs_of_cat(batch, i)
        avg_img = np.mean(imgs, axis=0)
        x, y = i % 5, (i // 5) * 2

        utils.plot_raw_img(avg_img.astype('int'), i, axs[y][x])
        utils.plot_RGB_hist(avg_img, axs[y + 1][x], hist_kws=hist_kws)

    plt.tight_layout()
    plt.show()

    """
    Histogram of RGB values from whole batch. Take all values into account
    instead of making an average image. But because whole dataset is too large,
    we use only random subsample of size 50 images from each category.
    """

    fig, axs = plt.subplots(2, 5, figsize=(15, 8))

    for i in range(10):
        imgs = utils.imgs_of_cat(batch, i)
        imgs = imgs[np.random.choice(imgs.shape[0], 50)]
        x, y = i % 5, i // 5

        axs[y][x].set_title(utils.get_lname(i))
        utils.plot_RGB_hist(imgs, axs[y][x])

    plt.tight_layout()
    plt.show()
