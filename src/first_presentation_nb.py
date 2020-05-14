### Image Recognition - Classifying images from CIFAR-10 dataset

##*Šimon Varga, Michal Barnišin*

##The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton and are available [here](https://www.cs.toronto.edu/~kriz/cifar.html). 

##### Loading data

import utils
import numpy as np

train_data, train_labels = utils.read_dataset()
test_data, test_labels = utils.read_test_batch()
labels = utils.read_meta()

#### Exploratory Analysis
##First, we examine the shape of the data as well as stored values.

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape:  {test_data.shape}")

## Divide et impera :D

display(train_data)
display(test_data)


##So we have 50,000 train images (image pre row), each row of the array stores a 32x32 colour image with 3 color channels (as specified on url above). The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
##The test batch contains exactly 1000 randomly-selected images from each class. As declared, the classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.
##Furthermore, all values are stored as uint8, which means, the range is always valid (0-255), and data don't contain any missing values.


display(train_labels)
min(train_labels), max(train_labels)


##Data are labeled with numbers in the range 0-9. String representations of labels are:


for i in range(0, 10):
    print(f"{i}\t{utils.get_label_name(i)}")


###### Classes distribution over train/test set


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


## Rozdeluj

test_distribution = [0 for i in range(10)]
for cat in range(0, 10):
    test_distribution[cat] = len(test_data[test_labels == cat])

frame = pd.DataFrame(data={
    "Category": [utils.get_label_name(i) for i in range(10)],
    "Pictures": test_distribution
})
sns.barplot(x="Category", y="Pictures", data=frame);


##From which we conclude, that all categories are equally present, so there is no need to weight the categories, or add/remove samples.

###### Pictures overview

##Now, we print 9 random pictures from all pictures, and 4 pictures per each category:


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

## Delete to save memory
del cat_images

# We also show average image here, as we will be using *SVM* to classify images, so that we can compares average image learnt by the model, and arithmetic mean.


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


##From which we can see, that color values, especially G and R channels, are similar for majority of images. Some histograms have large bin corresponding to RGB (255, 255, 255), which means there is probably white background.

###### Correlation

##Due to large size of data, we plot correlation matrix from random subset of size 500. # TODO


import matplotlib.pyplot as plt


sample = np.random.choice(len(train_data), 50, replace=False)
frame = pd.DataFrame(data=train_data[sample])
plt.matshow(frame.corr())
plt.show()

##The matrix appears to be divided to 9 zones, which is caused by 3 color channels in data, R, G, B in this order.

##Values near the main diagonal demostrate the property of real pictures, i.e. pixels are strongly correlated with nearby pixels. Furtheremore, the matrix suggests, that the images are symetric about the vertical, as for each sqaure, the submatrix is symetrix about main diagonal. As in each zone there is bright diagonal, the color channels are strongly correlated.


###### Outliers

##As the dataset doesn't really contain outliers, we plot the pictures from random selection of size 1000, which are the least likely to be in their class:

from sklearn.neighbors import LocalOutlierFactor


def detect_outliers(imgs, labels, test_size=1000):
    # Take only subset of data
    imgs = imgs[:10000]
    np.random.seed(0)
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

##We have found out that there are also images with "unnatural" backgrouds, such as bird with white (probably cropped) background.

#### Data Preprocessing

##First of all, we transform the data to RGB representation, i.e. to shape (?, 32, 32, 3), so that we can use functions from skimage library.

from preprocessing import batch_to_rgb


train_data = batch_to_rgb(train_data)
test_data = batch_to_rgb(test_data)
train_data.shape

##As mentioned above, there are no missing values nor values out of range. Because the range of values for RGB is 0-255, we first trasform values to floats and then scale to 0-1 interval.

train_data = train_data / 255.0
test_data = test_data / 255.0

##Now, each image has 32 * 32 * 3 = 3072 features. However from correlation matrix we saw, that there are many correlated features, so we will transform data even more. We will transform rgb to hsv, as we saw from histograms, that that might be most differentaiting, and remove saturation and value channels. Futhermore, we centralise the data and using PCA, we reduce dimensionality even further.

from skimage.color import rgb2hsv


pic = train_data[1545]  # to display origin picture
train_data = rgb2hsv(train_data)[:, :, :, 0]
test_data = rgb2hsv(test_data)[:, :, :, 0]
train_data.shape

##D & C

fig, ax = plt.subplots(1, 2, figsize=(8, 8))
ax[0].imshow(pic)
ax[0].set_title("Original")

ax[1].imshow(train_data[1545], cmap='hsv')
ax[1].set_title("Hue");

##Now we flatten the data, subtract mean and using PCA we will find the features to preserve 95% of variance.

train_data = train_data.reshape((train_data.shape[0], -1))
test_data = test_data.reshape((test_data.shape[0], -1))

train_data.shape, test_data.shape

## DC

mean_image = np.mean(train_data, axis=0)
train_data -= mean_image
test_data -= mean_image

train_data

## RaP

from sklearn.decomposition import PCA


pca = PCA(
    n_components=0.95,  # keep at least 95% of variance
    svd_solver='full',  # given by previous
    copy=True,          # apply the same transform to test set as well
).fit(train_data)


train_data = pca.transform(train_data)
test_data = pca.transform(test_data)


train_data.shape, test_data.shape

##So we were able to drop half of features to preserve 95% of variance. Finally, we split train_data to train and validation set.

from sklearn.model_selection import train_test_split


train_X, valid_X, train_y, valid_y = train_test_split(
    train_data, train_labels, test_size=1000, random_state=42
)

## <comment??> And delete previous to spare memory
del train_data
del train_labels

#### Baseline Model

##Either dummyClassifier or BayesianClassifier ??

#### Learning Models

##Definning and descibing the model, tuning hyperparameters....

#### Results

##Testing on test_data with test_labels, displaying graphs....

#### Conclusion

##Comparison of models, further improovments, naive models, state-of-the-art neural nets, why are some better than others, speed, data preprocessing....



