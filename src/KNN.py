from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import\
    GridSearchCV, cross_val_score, train_test_split
from sklearn.utils import resample
from sklearn.decomposition import PCA

from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog

from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np

from preprocessing import batch_to_rgb
from cache import cache
import utils

@cache
def gray_hog_prep(sample_X):
    gray_X = rgb2gray(batch_to_rgb(sample_X))
    hog_X = np.array(list(map(
        lambda img: hog(img, cells_per_block=(2, 2)), gray_X)))
    return hog_X
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                     metric_params=None, n_jobs=None, n_neighbors=10, p=1,
#                     weights='distance')
#0.47400000000000003

gray_hog_model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=10, p=1, weights='distance')


@cache
def hue_pca_prep(sample_X):
    hue_X = rgb2hsv(batch_to_rgb(sample_X) / 255.)[:, :, :, 0]
    hue_X = hue_X.reshape((sample_X.shape[0], -1))
    centered_X = hue_X - np.mean(hue_X, axis=0)
    pca = PCA(
        n_components=0.95,  # keep at least 95% of variance
        svd_solver='full',  # given by previous
        copy=True,          # apply the same transform to test set as well
    ).fit(centered_X)
    return pca.transform(centered_X)
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                     metric_params=None, n_jobs=None, n_neighbors=7, p=3,
#                     weights='distance')
#0.2596

hue_pca_model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=7, p=3, weights='distance')

@cache
def grid_search(train_X, train_y):
    param_grid = {
        'n_neighbors': [3, 5, 7, 10, 12],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3]
    }

    gscv = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        scoring='accuracy',
        n_jobs=-1,
        cv=5,
        verbose=20
    )

    gscv.fit(train_X, train_y)

    return (gscv.cv_results_, gscv.best_estimator_, gscv.best_score_)

def train_model(model, images, labels):
    if model == 'gray_hog':
        all_X = gray_hog_prep(images)
        return cross_val_score(gray_hog_model, all_X, labels,
                scoring='accuracy', cv=5, n_jobs=-1, verbose=20)
    elif model == 'hue_pca':
        all_X = hue_pca_prep(images)
        return cross_val_score(hue_pca_model, all_X, labels,
                scoring='accuracy', cv=5, n_jobs=-1, verbose=20)
    else:
        print("no such model")


if __name__ == '__main__':
    sns.set()

    print("load train images")

    all_images, all_labels = utils.read_dataset()

    do_grid_search = False

    if do_grid_search:
        print("choose subsample (10%) because of huge amount of images")

        sample_X, sample_y = resample(all_images, all_labels,
                replace=False, n_samples=5000, random_state=42,
                stratify=all_labels)

        print("preprocess")

        train_X = gray_hog_prep(sample_X)

        print("grid search")

        results, best_estimator, best_score = grid_search(train_X, sample_y)

        print("\nRESULTS:\n")

        print(results)
        print(best_estimator)
        print(best_score)
    else:
        print("training")

        scores = train_model('gray_hog', all_images, all_labels)
        print(scores)
        print(np.mean(scores))
