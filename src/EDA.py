"""
In this file we demonstrate the functions used for Exploratory
Data Analysis (EDA) of the CIFAR10 dataset.
"""
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import numpy as np
import utils
import graphs
import seaborn as sns
import pandas as pd
from typing import Dict


def plot_category_dist(labels, label_names):
    """
    Plots the distribution of values in *labels* as sns.barplot()
    :param labels: labels of the data, i.e. array with integers
    :param label_names: string representation of labels
    :return barplot with plotted data
    """
    bins = [np.count_nonzero(labels == cat) for cat in range(10)]
    frame = pd.DataFrame(data={
        "Category": label_names,
        "Pictures": bins
    })
    return sns.barplot(x="Category", y="Pictures", data=frame)


def plot_avg_imgs(batch: Dict, with_histogram: bool = True,
                  with_hsv: bool = False) -> np.array:
    """
    Plot average image and optionally also histogram of RGB values for each
    category.

    :param batch: dictionary with images and labels
    :param with_histogram: whether to plot also histograms
    :param with_hsv: transform avg image to hsv and plot histogram
    :return: np.array of average images (in raw ravel form)
    """
    import matplotlib.colors as colors
    d = 0
    if with_histogram:
        d += 1
    if with_hsv:
        d += 2
        
    im = plt.imread("../pics/hue.png")
    
    nrows = 2 * (1 + d)
    fig, axs = plt.subplots(nrows, 5, figsize=(15, 10))
    hist_kws = {'range': (50, 200)}
    avg_imgs = []
    
    for i in range(10):
        imgs = utils.imgs_of_cat(batch, i)
        avg_img = np.mean(imgs, axis=0)
        avg_imgs.append(avg_img)
        x, y = i % 5, i // 5
        y *= (1 + d)
    
        graphs.plot_raw_img(avg_img.astype('int'), i, axs[y][x])
        if with_histogram:
            y += 1
            graphs.plot_rgb_hist(avg_img, axs[y][x], hist_kws=hist_kws)
        if with_hsv:
            y += 1
            hsv_avg_img = colors.rgb_to_hsv(graphs.img_for_show(avg_img / 255))
            sns.distplot(hsv_avg_img[:, :, 0].reshape(-1), color='r',
                         label='hue', ax=axs[y][x])
            sns.distplot(hsv_avg_img[:, :, 1].reshape(-1), color='y',
                         label='saturation', ax=axs[y][x])
            sns.distplot(hsv_avg_img[:, :, 2].reshape(-1), color='b',
                         label='value', ax=axs[y][x])
            y += 1
            axs[y][x].imshow(im)
    return np.array(avg_imgs)


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


def plot_dendrogram(model, **kwargs) -> None:
    """Create linkage matrix and then plot the dendrogram
    See https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    :param model: Agglomerative clustering learned model
    :param **kwargs: Passed to scipy dendrogram ploting function
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)



if __name__ == '__main__':
    batch = utils.load_data_batch(1)

    # Print dataset size info about each category
    for i in range(10):
        imgs = utils.imgs_of_cat(batch, i)
        print(f"Category {i}: {utils.get_label_name(i)}")
        print(f"Size: {imgs.shape[0]}")
        print()

    # Find average image and plot histogram of RGB values for each category.
    avg_imgs = plot_avg_imgs(batch)
    plt.tight_layout()
    plt.show()

    # plot dendrogram
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(avg_imgs)

    plt.figure(figsize=(10, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, truncate_mode=None,
            labels=[utils.get_label_name(i) for i in range(10)])
    plt.xlabel("Category")
    plt.show()

    # Lets look at global histogram of each category. Use default size 50.
    plot_global_hist(batch)
    plt.tight_layout()
    plt.show()
