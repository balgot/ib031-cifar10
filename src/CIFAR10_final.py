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


## #### Classes distribution over train set and test set

import pandas as pd
import seaborn as sns

def plot_category_dist(labels):
    bins = [np.count_nonzero(labels == cat) for cat in range(10)]
    names = [utils.get_label_name(i) for i in range(10)]

    frame = pd.DataFrame(data={
        "Category": names,
        "Pictures": bins
    })
    return sns.barplot(x="Category", y="Pictures", data=frame);

plt.subplot(2, 1, 1)
plot_category_dist(train_labels).set_title("Train images")
plt.subplot(2, 1, 2)
plot_category_dist(test_labels).set_title("Test images")
plt.tight_layout()

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

#### Learning Models

##### Baseline Model

##Out of DummyClassifiers and Bayesian classifiers, we choose sklearn.DecisionTreeClassifier to serve as the baseline model. (DummyClassifier - 11%, BayesianGauss - 15%, DecisionTree - 25%).

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

##

"""grid_parameters = {
    "criterion": ["gini", "entropy"],
    "max_depth": [250, 100, 50],
    "min_samples_split": [50, 100, 150],
}


baseline_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid=grid_parameters,
    scoring='accuracy',
    n_jobs=-1,
    verbose=20,
    cv=5
)


baseline_search.fit(train_X, train_y)
baseline_search.best_params_"""

##

"""35.6min:

best_params_={'criterion': 'entropy', 'max_depth': 100, 'min_samples_split': 150}

baseline_search.cv_results_ = 
{'mean_fit_time': array([ 97.0174583 ,  91.89344964,  88.8256762 ,  95.62634368,
         88.08034205,  76.88998547,  89.45680666,  82.80917239,
         82.65973573, 353.10064287, 329.84735727, 319.94858952,
        359.98694668, 324.9900291 , 247.97087917, 235.17834697,
        208.78738151, 156.2808495 ]),
 'std_fit_time': array([ 2.86416142,  3.09920704,  3.70283776,  1.27372861,  3.22363977,
         0.6879136 ,  0.66807352,  0.91348156,  0.77450917,  3.12816108,
         2.28160371,  3.91188398,  3.60757506, 29.60802937,  5.13663667,
        16.56981693,  3.0810691 ,  0.95607979]),
 'mean_score_time': array([0.05862813, 0.0576756 , 0.05280013, 0.05199933, 0.05440049,
        0.04879999, 0.05679932, 0.04965858, 0.0593636 , 0.05120044,
        0.06131811, 0.05548034, 0.05370593, 0.05067997, 0.03370986,
        0.03357134, 0.03590431, 0.02789402]),
 'std_score_time': array([0.00652822, 0.00697919, 0.00587891, 0.00253047, 0.01175837,
        0.00299255, 0.010246  , 0.00347807, 0.01221577, 0.00160017,
        0.00989161, 0.00266799, 0.00320784, 0.0083324 , 0.0014658 ,
        0.00184262, 0.00252297, 0.00455124]),
 'param_criterion': masked_array(data=['gini', 'gini', 'gini', 'gini', 'gini', 'gini', 'gini',
                    'gini', 'gini', 'entropy', 'entropy', 'entropy',
                    'entropy', 'entropy', 'entropy', 'entropy', 'entropy',
                    'entropy'],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False],
        fill_value='?',
             dtype=object),
 'param_max_depth': masked_array(data=[250, 250, 250, 100, 100, 100, 50, 50, 50, 250, 250,
                    250, 100, 100, 100, 50, 50, 50],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False],
        fill_value='?',
             dtype=object),
 'param_min_samples_split': masked_array(data=[50, 100, 150, 50, 100, 150, 50, 100, 150, 50, 100, 150,
                    50, 100, 150, 50, 100, 150],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False],
        fill_value='?',
             dtype=object),
 'params': [{'criterion': 'gini', 'max_depth': 250, 'min_samples_split': 50},
  {'criterion': 'gini', 'max_depth': 250, 'min_samples_split': 100},
  {'criterion': 'gini', 'max_depth': 250, 'min_samples_split': 150},
  {'criterion': 'gini', 'max_depth': 100, 'min_samples_split': 50},
  {'criterion': 'gini', 'max_depth': 100, 'min_samples_split': 100},
  {'criterion': 'gini', 'max_depth': 100, 'min_samples_split': 150},
  {'criterion': 'gini', 'max_depth': 50, 'min_samples_split': 50},
  {'criterion': 'gini', 'max_depth': 50, 'min_samples_split': 100},
  {'criterion': 'gini', 'max_depth': 50, 'min_samples_split': 150},
  {'criterion': 'entropy', 'max_depth': 250, 'min_samples_split': 50},
  {'criterion': 'entropy', 'max_depth': 250, 'min_samples_split': 100},
  {'criterion': 'entropy', 'max_depth': 250, 'min_samples_split': 150},
  {'criterion': 'entropy', 'max_depth': 100, 'min_samples_split': 50},
  {'criterion': 'entropy', 'max_depth': 100, 'min_samples_split': 100},
  {'criterion': 'entropy', 'max_depth': 100, 'min_samples_split': 150},
  {'criterion': 'entropy', 'max_depth': 50, 'min_samples_split': 50},
  {'criterion': 'entropy', 'max_depth': 50, 'min_samples_split': 100},
  {'criterion': 'entropy', 'max_depth': 50, 'min_samples_split': 150}],
 'split0_test_score': array([0.25132653, 0.26408163, 0.27071429, 0.2505102 , 0.26489796,
        0.27071429, 0.25142857, 0.2644898 , 0.27071429, 0.23979592,
        0.2605102 , 0.26928571, 0.23959184, 0.2605102 , 0.26928571,
        0.23959184, 0.2605102 , 0.26918367]),
 'split1_test_score': array([0.23581633, 0.25346939, 0.26214286, 0.23622449, 0.25438776,
        0.26214286, 0.23602041, 0.25357143, 0.26214286, 0.23316327,
        0.25408163, 0.25877551, 0.23306122, 0.25418367, 0.25877551,
        0.23316327, 0.25408163, 0.25877551]),
 'split2_test_score': array([0.23336735, 0.25561224, 0.26      , 0.23306122, 0.25571429,
        0.26010204, 0.23316327, 0.25540816, 0.26010204, 0.24612245,
        0.25571429, 0.26877551, 0.24612245, 0.25571429, 0.26877551,
        0.24571429, 0.25571429, 0.26877551]),
 'split3_test_score': array([0.23255102, 0.25336735, 0.25959184, 0.23397959, 0.25377551,
        0.2594898 , 0.23397959, 0.25326531, 0.25959184, 0.23979592,
        0.25969388, 0.26602041, 0.23969388, 0.2594898 , 0.26602041,
        0.23969388, 0.25959184, 0.26602041]),
 'split4_test_score': array([0.23581633, 0.25306122, 0.25622449, 0.23540816, 0.25265306,
        0.25632653, 0.23561224, 0.25244898, 0.25632653, 0.23632653,
        0.24816327, 0.25979592, 0.23602041, 0.24826531, 0.25989796,
        0.23632653, 0.24826531, 0.25989796]),
 'mean_test_score': array([0.23777551, 0.25591837, 0.26173469, 0.23783673, 0.25628571,
        0.2617551 , 0.23804082, 0.25583673, 0.26177551, 0.23904082,
        0.25563265, 0.26453061, 0.23889796, 0.25563265, 0.26455102,
        0.23889796, 0.25563265, 0.26453061]),
 'std_test_score': array([0.00689977, 0.00418093, 0.00487366, 0.00643113, 0.00441807,
        0.00485285, 0.0067749 , 0.00443351, 0.00484349, 0.00431747,
        0.0044382 , 0.00443595, 0.00437392, 0.00436009, 0.0044143 ,
        0.00417316, 0.00438533, 0.00439255]),
 'rank_test_score': array([18,  8,  6, 17,  7,  5, 16,  9,  4, 13, 10,  2, 14, 11,  1, 14, 11,
         2])}"""

##

dtree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=100,
    min_samples_split=100,
)

dtree.fit(train_X, train_y)
pred = dtree.predict(valid_X)
print("Accuracy score\t", accuracy_score(valid_y, pred))
plot_confusion_matrix(dtree, valid_X, valid_y)

##

cheat = dtree.predict(test_data)
print("Accuracy score\t", accuracy_score(test_labels, cheat))
plot_confusion_matrix(dtree, test_data, test_labels)


##Definning and descibing the model, tuning hyperparameters....

#### Results

##Testing on test_data with test_labels, displaying graphs....

#### Conclusion

##Comparison of models, further improovments, naive models, state-of-the-art neural nets, why are some better than others, speed, data preprocessing....





