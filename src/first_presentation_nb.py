### Image Recognition - Classifying images from CIFAR-10 dataset

##*Šimon Varga, Michal Barnišin*

##The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton and are available [here](https://www.cs.toronto.edu/~kriz/cifar.html). 

##### Loading data

import utils
import numpy as np

def load_all_data():
    all_images, all_labels = [], []
    for batch_nb in range(1, 6):
        imgs, labels = utils.read_data_batch(batch_nb)
        all_images.append(imgs)
        all_labels.append(labels)
    del imgs, labels
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_images, all_labels


train_data, train_labels = load_all_data()
test_data, test_labels = utils.read_test_batch()
labels = utils.read_meta()

#### Exploratory data analysis

print(train_data.shape)
train_data

##So we have 50,000 train images = rows, each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

train_labels

##Data are labeled with numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data. The string representations of labels are:

for i in range(0, 1 + max(train_labels)):
    print(f"{i}\t{utils.get_label_name(i)}")

print(test_data.shape)
test_data

##The test batch contains exactly 1000 randomly-selected images from each class. 
##
##As declared, the classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

##Now, 9 random pictures follow, and 4 pictures from each category:

import matplotlib.pyplot as plt
import graphs

graphs.plot_random(train_data, train_labels, 3, 3,
            fontsize='small', sharey=True, sharex=True, figsize=(3, 3))
plt.tight_layout()
plt.show()

for i in range(len(labels)):
    label = labels[i].decode('utf-8')
    cat_images = train_data[train_labels == i]
    graphs.plot_random(cat_images, [i] * len(cat_images), 1, 4,
                       fontsize='small', sharey=True, sharex=True,
                       figsize=(8, 3))
    plt.tight_layout()
    plt.show()
    print(f"{label}: {len(cat_images)} entries")
del cat_images

##And average image for each category throughout the whole train set:

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

##From which we can see, that color values, especially G and R channels, are similar for majority of images.

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

##We have found out that there are also images with "unnatural" backgrouds.
##For example clear white background.
##Some histograms have large bin corresponding to RGB (255, 255, 255).

