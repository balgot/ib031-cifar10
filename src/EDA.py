from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import utils

"""
returns np.array of all images of specific category (e.i. label)
"""
def imgs_of_cat(batch, category):
    return batch[b'data'][np.array(batch[b'labels']) == category]

"""
creates new image prepared to show from raw data from batch
"""
def img_for_show(raw_img):
    return raw_img.reshape((32, 32, 3), order='F').swapaxes(0, 1)


if __name__ == '__main__':
    batch = utils.load_data_batch(1)
    meta = utils.load_meta()
    labels = meta[b'label_names']

    """
    Print dataset size info about each category
    """
    for i in range(10):
        imgs = imgs_of_cat(batch, i)
        print("Category {}: {}".format(i, labels[i]))
        print("Size: {}".format(imgs.shape[0]))
        print()

    """
    Find average image and plot histogram of RGB values for each category
    """

    fig, axs = plt.subplots(4, 5, figsize=(15, 10))
    hist_kws = { 'range' : (50, 200) }

    for i in range(10):
        imgs = imgs_of_cat(batch, i)
        avg_img = np.mean(imgs, axis=0)
        R, G, B = avg_img[:1024], avg_img[1024:2048], avg_img[2048:]
        x, y = i % 5, (i // 5) * 2

        axs[y][x].imshow(img_for_show(avg_img.astype('int')))
        axs[y][x].set_title(labels[i])
        axs[y][x].get_yaxis().set_visible(False)
        axs[y][x].get_xaxis().set_visible(False)
        sns.distplot(R, hist_kws=hist_kws, color='red',   ax=axs[y + 1][x])
        sns.distplot(G, hist_kws=hist_kws, color='green', ax=axs[y + 1][x])
        sns.distplot(B, hist_kws=hist_kws, color='blue',  ax=axs[y + 1][x])

    plt.show()

    """
    Histogram of RGB values from whole batch. Take all values into account
    instead of making an average image. But because whole dataset is too large,
    we use only random subsample of size 100 images from each category.
    """

    fig, axs = plt.subplots(2, 5, figsize=(15, 8))
    hist_kws = { 'range' : (0, 255) }

    for i in range(10):
        imgs = imgs_of_cat(batch, i)
        imgs = imgs[np.random.choice(imgs.shape[0], 100)]
        R = imgs[:,     :1024].reshape(-1)
        G = imgs[:, 1024:2048].reshape(-1)
        B = imgs[:, 2048:    ].reshape(-1)
        x, y = i % 5, i // 5

        axs[y][x].set_title(labels[i])
        sns.distplot(R, hist_kws=hist_kws, color='red',   ax=axs[y][x])
        sns.distplot(G, hist_kws=hist_kws, color='green', ax=axs[y][x])
        sns.distplot(B, hist_kws=hist_kws, color='blue',  ax=axs[y][x])

    plt.show()
