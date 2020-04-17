from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

"""
Default path to untared image data
"""
CIFAR_PATH = '../dataset'

"""
Loaded in this way, each of the batch files contains a dictionary with the
following elements:

* data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a
  32x32 colour image. The first 1024 entries contain the red channel values,
  the next 1024 the green, and the final 1024 the blue. The image is stored
  in row-major order, so that the first 32 entries of the array are the red
  channel values of the first row of the image.
* labels -- a list of 10000 numbers in the range 0-9. The number at index i
  indicates the label of the ith image in the array data.

The dataset contains another file, called batches.meta. It too contains
a Python dictionary object. It has the following entries: 
* label_names -- a 10-element list which gives meaningful names to the numeric
  labels in the labels array described above. For example,
  label_names[0] == "airplane", label_names[1] == "automobile", etc.
"""

def unpickle(file):
    """
    Loads data from file as is suggested on original cifar website
    :param file: file to read
    :return: dict (format specified in global docs)
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_meta(path=CIFAR_PATH):
    return unpickle(path + '/batches.meta')

def load_data_batch(i, path=CIFAR_PATH):
    return unpickle(path + '/data_batch_' + str(i))

def load_test_batch(path=CIFAR_PATH):
    return unpickle(path + '/test_batch')

def get_lname(label):
    """
    :param label: integer image label
    :return: corresponding string label name
    """
    return load_meta()[b'label_names'][label].decode('utf-8')

def imgs_of_cat(batch, category):
    """
    :param batch: From where to obtain images
    :param category: Images of what category to obtain
    :return: np.array of all images of specific category (e.i. label) 
    """
    return batch[b'data'][np.array(batch[b'labels']) == category]

def img_for_show(raw_img):
    """
    Creates new image prepared to show from raw data from batch
    :param raw_img: 1D format of image
    :return: 3D format (y, x, rgb) ready for pyplot's imshow
    """
    return raw_img.reshape((32, 32, 3), order='F').swapaxes(0, 1)

def plot_raw_img(raw_img, label, ax, fontsize='medium'):
    """
    Plots single image on specified ax (see Axes).
    :param raw_img: format as in batch (1D array [R, G, B])
    :param label: integer label (None for no title)
    :param ax: ax to plot on
    :param fontsize: passed to ax.set_title
    """
    ax.imshow(img_for_show(raw_img))
    if label is not None:
        ax.set_title(f"{label}: {get_lname(label)}", fontsize=fontsize)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

def plot_RGB_hist(raw_imgs, ax, **kwargs):
    """
    Plots histograms of image(s)'s R, G, B amount using seaborn
    :param raw_imgs: array of images (1D format) or just one such image
    :param ax: ax on which to plot
    :param **kwargs: passed to seaborn.distplot
    """
    R = raw_imgs.reshape((-1, 3072))[:,     :1024].reshape(-1)
    G = raw_imgs.reshape((-1, 3072))[:, 1024:2048].reshape(-1)
    B = raw_imgs.reshape((-1, 3072))[:, 2048:    ].reshape(-1)
    sns.distplot(R, color='red',   ax=ax, **kwargs)
    sns.distplot(G, color='green', ax=ax, **kwargs)
    sns.distplot(B, color='blue',  ax=ax, **kwargs)


def plot_random(imgs, labels, nrows=1, ncols=1, fontsize='medium', **kwargs):
    """
    Plots random images from batch using matplotlib pyplot.
    User should typically call pyplot.show() after this function

    :param imgs: images in form 1D array [Rvalue, Gvalues, Bvalues]
    :param labels: labels (numerical) attached to images
    :param nrows: number of rows of images
    :param ncols: number of cols of images
    :param fontsize: passed to pyplot.Axes.set_title
    :param **kwargs: passed to pyplot.subplots
    """
    fig, axs = plt.subplots(nrows, ncols, **kwargs)

    for ax in axs.reshape(-1):
        random = np.random.randint(0, imgs.shape[0])
        img = imgs[random]
        label = labels[random]
        plot_raw_img(img, label, ax, fontsize=fontsize)
