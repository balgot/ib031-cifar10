TARGETS := test_nb.ipynb first_presentation_nb.ipynb CIFAR10.ipynb

all: $(TARGETS)

$(TARGETS):
	jupytext --sync $@

clean:
	rm -f $(TARGETS)

.PHONY: all $(TARGETS) clean
