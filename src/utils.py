import numpy as np
from typing import Dict, Any, Tuple
import cache


# Default path to un-tarred image data
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


def unpickle(path: str) -> Dict:
    """
    Loads data from file as is suggested on original cifar website

    :param path: name of file to read
    :return: dict (format specified in global docs)
    """
    import pickle
    with open(path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


@cache.cache_on_the_fly
def load_meta(path: str = CIFAR_PATH, filename: str = '/batches.meta') -> Dict:
    """
    Load meta data from pickled file.

    :param path: path to file-folder
    :param filename: name of file with pickled meta data
    :return: dictionary with metadata
    """
    return unpickle(path + filename)


def load_data_batch(i: int, path: str = CIFAR_PATH, filename: str = '/data_batch_') -> Dict:
    """
    Load i-th data batch.

    :param i: which batch to load
    :param path: path to folder with file
    :param filename: name of file with pickled data
    :return: dictionary with loaded data
    """
    return unpickle(path + filename + str(i))


def load_test_batch(path: str = CIFAR_PATH, filename: str = '/test_batch') -> Dict:
    """
    Load test batch.

    :param path: path to folder with file
    :param filename: name of file with pickled data
    :return: dictionary with loaded data
    """
    return unpickle(path + filename)


def read_data_batch(i: int, path: str = CIFAR_PATH) -> Tuple[Any, np.ndarray]:
    """
    Load and process data-batch file.

    :param i: which batch to load
    :param path: path to folder with file
    :return: tuple (imgs, labels) of np.arrays (note: integer labels)
    """
    batch = load_data_batch(i, path)
    return batch[b'data'], np.array(batch[b'labels'])

def read_dataset(path: str = CIFAR_PATH) -> Tuple[Any, np.ndarray]:
    """
    Load and process (concatenate) all data-batch files.

    :param path: path to folder with file
    :return: tuple (imgs, labels) of np.arrays (note: integer labels)
    """
    all_images, all_labels = [], []
    for i in range(1, 6):
        imgs, labels = utils.read_data_batch(i)
        all_images.append(imgs)
        all_labels.append(labels)
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_images, all_labels

def read_test_batch(path: str = CIFAR_PATH) -> Tuple[Any, np.ndarray]:
    """
    Load and process test file.

    :param path: path to folder with file
    :return: tuple (imgs, labels) of np.arrays (note: integer labels)
    """
    batch = load_test_batch(path)
    return batch[b'data'], np.array(batch[b'labels'])


def read_meta(path: str = CIFAR_PATH) -> np.ndarray:
    """
    Loads and returns CIFAR10 label names in order, i.e.
    to label 0, 0-th string name corresponds.

    :param path: path to meta folder
    :return: np.array of label names (of type np.bytes_)
    """
    return np.array(load_meta(path)[b'label_names'])


def get_label_name(label: int) -> str:
    """
    :param label: integer image label
    :return: corresponding string label name
    """
    return read_meta()[label].decode('utf-8')


def imgs_of_cat(batch: Dict, category: int) -> np.ndarray:
    """
    :param batch: From where to obtain images
    :param category: Images of what category to obtain
    :return: np.array of all images of specific category (e.i. label) 
    """
    return batch[b'data'][np.array(batch[b'labels']) == category]
