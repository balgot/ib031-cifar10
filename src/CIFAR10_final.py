## <div style="max-width: 900px; position: relative; margin: auto"><h1 style="text-align: center">Image Recognition – Classifying images from CIFAR-10 dataset</h1><p style="text-align: right; padding-right: 3em; color: #8c8c8c;"><i>Šimon Varga, Michal Barnišin</i></p></div>

## The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton and are available [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## ### Loading Data

import utils
import numpy as np

train_data, train_labels = utils.read_dataset()
test_data, test_labels = utils.read_test_batch()
labels = utils.read_meta()

## ## Exploratory Analysis

## Now, the dimensions and samples from data follow.

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape:  {test_data.shape}")


display(train_data)
display(test_data)

## So we have 50,000 train images (image pre row), each row of the array stores a 32x32 colour image (as specified on url above). The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

## The test batch contains exactly 1000 randomly-selected images from each class. As declared, the classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

## Furthermore, all values are stored as uint8, which means, the range is always valid (0-255), and data don't contain any missing values.

display(train_labels)
min(train_labels), max(train_labels)

## Data are labeled with numbers in the range 0-9. String representations of labels are:

for i in range(0, 10):
    print(f"{i}\t{utils.get_label_name(i)}")


## #### Classes distribution over train set

import pandas as pd
import seaborn as sns

train_distribution = [0 for i in range(10)]
for cat in range(0, 10):
    train_distribution[cat] = len(train_data[train_labels == cat])

frame = pd.DataFrame(data={
    "Category": [utils.get_label_name(i) for i in range(10)],
    "Pictures": train_distribution
})
sns.barplot(x="Category", y="Pictures", data=frame);

## and test set

test_distribution = [0 for i in range(10)]
for cat in range(0, 10):
    test_distribution[cat] = len(test_data[test_labels == cat])

frame = pd.DataFrame(data={
    "Category": [utils.get_label_name(i) for i in range(10)],
    "Pictures": test_distribution
})
sns.barplot(x="Category", y="Pictures", data=frame);

## From which we conclude, that all categories are equally present, so there is no need to weight the categories, or add/remove samples.

## #### Pictures overview

## Now, we print 9 random pictures from all pictures, and 4 pictures per each category:

import matplotlib.pyplot as plt
import graphs

# Prints 3x3 pictures randomly
graphs.plot_random(train_data, train_labels, 3, 3,
                   fontsize='small', sharey=True, sharex=True, figsize=(3, 3))
plt.tight_layout()
plt.show()

# 4 Pictures per category
for i in range(len(labels)):
    label = labels[i].decode('utf-8')
    cat_images = train_data[train_labels == i]
    graphs.plot_random(cat_images, [i] * len(cat_images), 1, 4,
                       fontsize='small', sharey=True, sharex=True,
                       figsize=(8, 3))
    plt.tight_layout()
    plt.show()

# Delete to save memory
del cat_images

## We also show average image here, as we use this (?) in one of our models

import EDA

batch = {
    b'data': train_data,
    b'labels': train_labels
}

EDA.plot_avg_imgs(batch, with_histogram=True, with_hsv=True)
plt.tight_layout()
plt.show()

EDA.plot_global_hist(batch, sample_size=int(len(train_data) ** 0.25))
plt.tight_layout()

## From which we can see, that color values, especially G and R channels, are similar for majority of images. Some histograms have large bin corresponding to RGB (255, 255, 255), which means there is probably white background.

## #### Correlation

## Due to large size of data, we plot correlation matrix from random subset of size 100.

import matplotlib.pyplot as plt


def transform_to_pd(data: np.ndarray) -> pd.DataFrame:
    """
    Returns pd.DataFrame from data, given the first
    dimension is samples, second attribute values.

    :param data np.ndarray of pictures
    :return pd.DataFrame with the data
    """
    return pd.DataFrame(data=data)


sample = np.random.choice(len(train_data), 100, replace=False)
frame = transform_to_pd(train_data[sample])
plt.matshow(frame.corr())
plt.show()

## The matrix appears to be divided to 9 zones, which is caused by 3 color channels in data, R, G, B in this order.

## Values near the main diagonal demostrate the property of real pictures, i.e. pixels are strongly correlated with nearby pixels. Furtheremore, the matrix suggests, that the images are symetric about the vertical, as for each sqaure, the submatrix is symetrix about main diagonal.

## #### Outliers

## As the dataset doesn't really contain outliers, we plot the pictures from random selection of size 1000, which are the least likely to be in their class:

from sklearn.neighbors import LocalOutlierFactor


def detect_outliers(imgs, labels, test_size=1000):
    # Take only subset of data
    imgs = imgs[:10000]
    sample = np.random.choice(len(imgs), test_size, replace=False)
    imgs = imgs[sample]
    labels = labels[sample]

    # Scale data. For images it is simple, every pixel is in range 0-255.
    imgs = imgs / 255

    # Find Outliers using LOF
    LOF = LocalOutlierFactor()
    outliers = np.where(LOF.fit_predict(imgs) == -1)[0]

    for i in range(len(labels)):
        cat_images = train_data[outliers][train_labels[outliers] == i]
        if len(cat_images) == 0:
            continue
        print(utils.get_label_name(i))
        graphs.plot_images(cat_images)
        plt.tight_layout()
        plt.show()


detect_outliers(train_data, train_labels, test_size=1000)

## We have found out that there are also images with "unnatural" backgrouds.

## ## Data Preprocessing

## As mentioned above, there are no missing values nor values out of range. Because the range is 0-255, we first trasform values to floats and then scale to 0-1 interval.

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data = train_data / 255
test_data = test_data / 255

train_data[0]

## ## Learning Models

## ## Results

## ## Baseline Model

## ## Conclusion




