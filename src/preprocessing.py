import numpy as np
import utils
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog
from skimage import filters
import matplotlib.pyplot as plt


def batch_to_rgb(images: np.ndarray) -> np.ndarray:
    """
    Given loaded images from CIFAR-10 dataset (i.e. 32x32 values
    of red, then green and blue), returns same set of images with
    color channel being the last, i.e. batch_to_rgb[n][y][x] returns
    3-value array with r, g, b of pixel (x, y) of n-th image.

    :param images: CIFAR-images in default format
    :return: same images with transformed colors to rgb

    Time: 0.0s on whole set
    """
    return images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)


def rgb_to_gray(images: np.ndarray) -> np.ndarray:
    """
    Converts rgb images to gray scale.
    :param images: rgb images of shape [.., .., .., 3]
    :return: array with channel dimension removed

    Time: 1.081s on whole set (commented 1.8s)
    """
    # return np.dot(images[...,:3], [0.2125, 0.7154, 0.0721])
    return rgb2gray(images)


def rgb_to_hsv(images: np.ndarray) -> np.ndarray:
    """
    Converts rgb images to HSV color space.
    :param images: rgb images of shape [?, .., .., 3]
    :return: array of images in HSV, each with shape [?, .., .., 3]

    Time: 0.0s on whole set
    """
    if len(images.shape) == 4:
        return np.array(map(rgb2hsv, images))
    return rgb2hsv(images)


def brightness_norm(batch: np.ndarray) -> np.ndarray:
    """
    Simple brightness normalization. For any array that satisfies
    shape[-1] == 3, divides values by maximum value along this last (-1) axis.

    :param batch: Usually rgb image/s of shape [?...?, 3]
    :return: normalized image/s of the same shape
    """
    maxes = batch.max(axis=-1, initial=0)
    maxes_stack = np.stack((maxes, maxes, maxes), axis=-1)
    return np.divide(batch, maxes_stack,
                     out=np.zeros_like(batch, dtype='float32'),
                     where=maxes_stack != 0)


def demo(X: np.ndarray, ax: np.ndarray) -> None:
    """
    Shows results of various preprocessing methods.

    :param X: np.array (shape [.., .., .., 3]) of images in rgb shape
    :param ax: flattened axis (usually from pyplot.subplots(...)[2])
    """
    pic = X[1545]

    ax[0].imshow(pic)
    ax[0].set_title("Original")
    print(f"Original = {pic.shape}")

    gray = rgb_to_gray(pic)
    ax[1].imshow(gray, cmap=plt.cm.gray)
    ax[1].set_title("Gray")
    print(f"Gray = {gray.shape}")

    # Now to transform whole array
    grays = rgb_to_gray(X)
    ax[2].imshow(grays[1545], cmap=plt.cm.gray)
    ax[2].set_title("Gray - array")

    hsv = rgb_to_hsv(pic)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    ax[3].imshow(hue, cmap='hsv')
    ax[3].set_title("Hue")
    ax[4].imshow(sat, cmap='hsv')
    ax[4].set_title("Saturation")
    ax[5].imshow(val, cmap='hsv')
    ax[5].set_title("Value")
    print(f"HSV = {hsv.shape}")

    # HOG
    fd, hog_image = hog(pic, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    ax[6].imshow(hog_image, cmap=plt.cm.gray)
    ax[6].set_title("Hog")
    print(f"Hog = {fd.shape}")  # 128 = (32 * 32) / (8 * 8) * orientations[=8]

    # hog detailed
    fd_det, hog_det = hog(pic, pixels_per_cell=(4, 4), cells_per_block=(1, 1),
                          visualize=True, multichannel=True)
    ax[7].imshow(hog_det, cmap=plt.cm.gray)
    ax[7].set_title("Hog detailed")
    # print(fd_det.shape)  # 576 = (32 * 32) / (4 * 4) * orientations[=9]

    # hog more detailed
    _, hog_det_ = hog(pic, pixels_per_cell=(2, 2), cells_per_block=(1, 1),
                      visualize=True, multichannel=True)
    ax[8].imshow(hog_det_, cmap=plt.cm.gray)
    ax[8].set_title("Hog more detailed")

    # edges detection
    sobel = filters.sobel(rgb_to_gray(pic))
    ax[9].imshow(sobel)
    ax[9].set_title("Sobel (edges)")
    print(f"Sobel = {sobel.shape}")

    roberts = filters.roberts(rgb_to_gray(pic))
    ax[10].imshow(roberts)
    ax[10].set_title("Roberts (edges)")
    print(f"Roberts = {roberts.shape}")

    prewitt = filters.prewitt(rgb_to_gray(pic))
    ax[11].imshow(prewitt)
    ax[11].set_title("Prewitt (edges)")
    print(f"Prewitt = {prewitt.shape}")
    print(prewitt)


if __name__ == "__main__":
    X, y = utils.read_data_batch(1)
    fig, axes = plt.subplots(8, 3, figsize=(8, 8))
    ax = axes.ravel()

    # !NORMALISE to range <0, 1>
    X = batch_to_rgb(X) / 255.0
    demo(X, ax[:3*4])

    print("\nAnd now after brightness normalization\n")
    X = brightness_norm(X)
    demo(X, ax[3*4:])

    fig.tight_layout()
    plt.show()
