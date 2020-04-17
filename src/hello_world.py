from matplotlib import pyplot as plt
import utils

if __name__ == '__main__':
    print("hello world")
    imgs, labels = utils.read_data_batch(1)
    utils.plot_random(imgs, labels, 3, 3,
            fontsize='small', sharey=True, sharex=True, figsize=(3, 3))
    plt.tight_layout()
    plt.show()
