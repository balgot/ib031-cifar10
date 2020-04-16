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
        axs[y][x].set_title(meta[b'label_names'][i])
        axs[y][x].get_yaxis().set_visible(False)
        axs[y][x].get_xaxis().set_visible(False)
        sns.distplot(R, hist_kws=hist_kws, color='red',   ax=axs[y + 1][x])
        sns.distplot(G, hist_kws=hist_kws, color='green', ax=axs[y + 1][x])
        sns.distplot(B, hist_kws=hist_kws, color='blue',  ax=axs[y + 1][x])


    plt.show()
