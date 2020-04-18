from matplotlib import pyplot as plt
import utils
import graphs


# Simple demo to demonstrate plotting graphs
if __name__ == '__main__':
    # Load first data-batch
    imgs, labels = utils.read_data_batch(1)
    # Plot 3 * 3 = 9 random images
    graphs.plot_random(imgs, labels, 3, 3, fontsize='small',
                       sharey=True, sharex=True, figsize=(3, 3))
    plt.tight_layout()
    plt.show()
