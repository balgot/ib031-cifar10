# ib031-cifar10

## Usage

### Setup

After downloading probably You want to see notebooks. To create them:
1. Navigate to "src" directory
2. run `jupytext --to notebook [script]` where script is the name of script
   You want to generate notebook from. At this time it may be one of:
   * test_nb.py
   * first_presentation_nb.py
   * CIFAR10.py
   To see this info actualized, run `grep TARGETS Makefile | head -1`
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
