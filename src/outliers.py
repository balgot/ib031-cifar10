from sklearn.neighbors import LocalOutlierFactor
from matplotlib import pyplot as plt
import numpy as np
import utils

if __name__ == '__main__':
    batch = utils.load_data_batch(1)
    meta = utils.load_meta()

    """
    Use just a subset - it is very time consuming and we just want to have
    overview of how bad images could be. It is dataset made for learning
    image classification, so there are no "real" outliers. For example
    whole image clear.
    """
    sample = np.random.choice(10000, 500)
    imgs = batch[b'data'][sample]
    labels = np.array(batch[b'labels'])[sample]

    """
    Scale data. For images it is simple, every pixel is in range 0-255.
    """
    imgs = imgs / 255

    LOF = LocalOutlierFactor()
    outliers = np.where(LOF.fit_predict(imgs) == -1)[0]
    print(outliers)

    for i in outliers:
        plt.imshow(utils.img_for_show(imgs[i]))
        plt.title(meta[b'label_names'][labels[i]])
        plt.show()

    """
    We have found out that there are also images with "unnatural" backgrouds.
    For example clear white background. It can bee seen alsi from EDA.py.
    Some histograms have large bin corresponding to RGB (255, 255, 255).
    """
