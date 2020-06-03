TARGETS := CIFAR10.ipynb

all: $(TARGETS)

$(TARGETS):
	jupytext --sync $@

clean:
	rm -f $(TARGETS)

.PHONY: all $(TARGETS) clean
