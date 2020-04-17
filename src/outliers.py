from sklearn.neighbors import LocalOutlierFactor
from matplotlib import pyplot as plt
import numpy as np
import utils

if __name__ == '__main__':
    """
    Use just a subset - it is very time consuming and we just want to have
    overview of how bad images could be. It is dataset made for learning
    image classification, so there are no "real" outliers. For example
    whole image clear.
    """
    imgs, labels = utils.read_data_batch(1)
    sample = np.random.choice(10000, 500)
    imgs = imgs[sample]
    labels = labels[sample]

    """
    Scale data. For images it is simple, every pixel is in range 0-255.
    """
    imgs = imgs / 255

    """
    Find outliers using LOF
    """

    LOF = LocalOutlierFactor()
    outliers = np.where(LOF.fit_predict(imgs) == -1)[0]
    print(outliers)

    """
    Lets see found outliers.
    """

    for i in outliers:
        plt.figure(figsize=(1, 1))
        utils.plot_raw_img(imgs[i], labels[i], plt.gca())
        plt.tight_layout()
        plt.show()

    """
    We have found out that there are also images with "unnatural" backgrouds.
    For example clear white background. It can bee seen also from EDA.py.
    Some histograms have large bin corresponding to RGB (255, 255, 255).
    """
