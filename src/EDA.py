from matplotlib import pyplot as plt
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

    fig, axs = plt.subplots(4, 10, figsize=(15, 10))

    for i in range(10):
        imgs = imgs_of_cat(batch, i)
        avg_img = np.sum(imgs, axis=0) / imgs.shape[0]
        R, G, B = avg_img[:1024], avg_img[1024:2048], avg_img[2048:]

        axs[0][i].imshow(img_for_show(avg_img.astype('int')))
        axs[0][i].set_title(meta[b'label_names'][i])
        axs[0][i].get_yaxis().set_visible(False)
        axs[0][i].get_xaxis().set_visible(False)
        axs[1][i].hist(R, bins=50, range=(0, 255), color='red')
        axs[2][i].hist(G, bins=50, range=(0, 255), color='green')
        axs[3][i].hist(B, bins=50, range=(0, 255), color='blue')


    plt.show()
