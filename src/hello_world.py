from matplotlib import pyplot as plt
import numpy as np
import utils

if __name__ == '__main__':
    print("hello world")

    batch = utils.load_data_batch(1)
    meta = utils.load_meta()

    fig, axs = plt.subplots(3, 3, sharey=True, sharex=True, figsize=(6, 6))

    for ax in axs.reshape(9):
        random = np.random.randint(0, 10001)
        img = batch[b'data'][random]
        label = batch[b'labels'][random]

        img = img.reshape((32, 32, 3), order='F').swapaxes(0, 1)
        title = "{}: {}".format(label, meta[b'label_names'][label])

        ax.imshow(img)
        ax.set_title(title)

    plt.show()
