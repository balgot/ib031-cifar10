# ib031-cifar10

This project was created as the assignment solution for the IB031 subject on FI MU in Brno, Czech Republic.

In this project, we will try to classify CIFAR-10 dataset images, using a simple classifier, by which we mean not neural networks. Our aim is to maximize accuracy. Therefore and with respect to the nature of this dataset, we will use the k-Nearest Neighbours model and Support Vector Machines.

We will try two different approaches to pre-processing, and we will evaluate the results for both models. Eventually, we will compare the models with respect to accuracy (mainly), and also the time needed for predictions and training.

## Results

After preprocessing by HOG descriptors and HUE transformation, we conclude that for all models, the HOG transformation is faster and more accurate, with baseline model (Decision Tree)  achieving 31% accuracy on the test set, KNN model 55% and the Bagging Ensemble with SVM achieved 59%, however slight overfitting was present.

## Usage

### Setup

After downloading probably you can find the final notebook in "src" directory. However, you can recreate the available notebooks from the source files using *jupytext*:
1. Navigate to "src" directory
2. run `jupytext --to notebook [script]` where script is the name of script
   You want to generate notebook from. At this time it may be one of:
   * CIFAR10.py (contains all the code, still being updated to match *src/CIFAR10.ipynb*
3. Then You can run it (or here is a shorthand:
   `jupytext --to notebook --execute [script]`)
4. If You are a developer and already have the notebook computed, it (should)
   be enough to just rename it.

### Making changes

You can make changes both in notebook or script. But to synchronize
them run `jupyter --sync [notebook]` or just `make [notebook]`
(Try typing "make " and then <tab> should iterate over all notebooks)

### Adding notebooks

To add new notebook to git tracking:
1. `jupytext --set-formats ipynb,py [notebook]`
2. Add `"jupytext": { "notebook_metadata_filter": "all" }` to notebook metadata 
   (Edit -> Edit Notebook Metadata)
